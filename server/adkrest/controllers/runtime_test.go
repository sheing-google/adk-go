// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package controllers

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"iter"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/server/adkrest/internal/models"
	"google.golang.org/adk/session"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"
)

type fakeAgent struct {
	agent.Agent
	err error
}

func (a *fakeAgent) Run(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		if a.err != nil {
			yield(nil, a.err)
		}
	}
}

func (a *fakeAgent) Name() string {
	return "fakeAgent"
}

func (a *fakeAgent) SubAgents() []agent.Agent {
	return nil
}

type fakeEvents struct {
	session.Events
}

func (e *fakeEvents) Len() int                 { return 0 }
func (e *fakeEvents) At(i int) *session.Event { return nil }

type fakeSession struct {
	session.Session
}

func (s *fakeSession) ID() string             { return "testSession" }
func (s *fakeSession) AppName() string        { return "testApp" }
func (s *fakeSession) UserID() string         { return "testUser" }
func (s *fakeSession) Events() session.Events { return &fakeEvents{} }
func (s *fakeSession) State() session.State   { return nil }
func (s *fakeSession) UpdatedAt() time.Time   { return time.Now() }

type fakeSessionService struct {
	session.Service
	session session.Session
	err     error
}

func (s *fakeSessionService) Get(ctx context.Context, req *session.GetRequest) (*session.GetResponse, error) {
	if s.err != nil {
		return nil, s.err
	}
	if s.session == nil {
		return &session.GetResponse{Session: &fakeSession{}}, nil
	}
	return &session.GetResponse{Session: s.session}, nil
}

func (s *fakeSessionService) AppendEvent(ctx context.Context, sesh session.Session, event *session.Event) error {
	return nil
}

type fakeAgentLoader struct {
	agent.Loader
	agent agent.Agent
	err   error
}

func (l *fakeAgentLoader) LoadAgent(appName string) (agent.Agent, error) {
	if l.err != nil {
		return nil, l.err
	}
	return l.agent, nil
}

func TestRunSSEHandler(t *testing.T) {
	testCases := []struct {
		name               string
		mockAgent          agent.Agent
		mockSession        session.Session
		mockSessionErr     error
		mockAgentLoaderErr error
		wantStatusCode     int
		wantBodyContains   string
		expectPanic        bool
	}{
		{
			name:             "Runner returns an error",
			mockAgent:        &fakeAgent{err: errors.New("LLM error")},
			mockSession:      &fakeSession{},
			wantStatusCode:   http.StatusOK,
			wantBodyContains: "Error while running agent: LLM error",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					if !tc.expectPanic {
						t.Errorf("The code panicked when it should not have: %v", r)
					}
				} else {
					if tc.expectPanic {
						t.Error("The code did not panic when it should have.")
					}
				}
			}()

			sessionService := &fakeSessionService{
				session: tc.mockSession,
				err:     tc.mockSessionErr,
			}
			agentLoader := &fakeAgentLoader{
				agent: tc.mockAgent,
				err:   tc.mockAgentLoaderErr,
			}
			controller := NewRuntimeAPIController(sessionService, agentLoader, nil, 1*time.Second)

			handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if err := controller.RunSSEHandler(w, r); err != nil {
					// The handler now returns nil on success and after flushing the error,
					// so we don't expect an error here in the test case.
					t.Errorf("RunSSEHandler returned an unexpected error: %v", err)
				}
			})

			server := httptest.NewServer(handler)
			defer server.Close()

			runAgentRequest := models.RunAgentRequest{
				AppName:    "testApp",
				UserId:     "testUser",
				SessionId:  "testSession",
				NewMessage: genai.Content{},
				Streaming:  true,
			}

			body, _ := json.Marshal(runAgentRequest)
			resp, err := http.Post(server.URL, "application/json", bytes.NewReader(body))
			if err != nil {
				t.Fatalf("failed to send request to test server: %v", err)
			}
			defer resp.Body.Close()

			if resp.StatusCode != tc.wantStatusCode {
				t.Errorf("handler returned wrong status code: got %d want %d", resp.StatusCode, tc.wantStatusCode)
			}

			respBody, _ := io.ReadAll(resp.Body)
			if !strings.Contains(string(respBody), tc.wantBodyContains) {
				t.Errorf("handler returned unexpected body: got %q, want to contain %q", string(respBody), tc.wantBodyContains)
			}
		})
	}
}

func TestRunHandler(t *testing.T) {
	testCases := []struct {
		name               string
		reqBody            models.RunAgentRequest
		mockSession        session.Session
		mockSessionErr     error
		mockAgent          agent.Agent
		mockAgentLoaderErr error
		wantStatusCode     int
		wantRespBody       []models.Event
		wantErr            string
	}{
		{
			name: "successful run",
			reqBody: models.RunAgentRequest{
				AppName:    "testApp",
				UserId:     "testUser",
				SessionId:  "testSession",
				NewMessage: genai.Content{},
			},
			mockSession:    &fakeSession{},
			mockAgent:      &fakeAgent{},
			wantStatusCode: http.StatusOK,
			wantRespBody:   []models.Event{},
		},
		{
			name: "session not found",
			reqBody: models.RunAgentRequest{
				AppName:   "testApp",
				UserId:    "testUser",
				SessionId: "testSession",
			},
			mockSessionErr: errors.New("not found"),
			wantStatusCode: http.StatusNotFound,
			wantErr:        "failed to get session: not found",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			sessionService := &fakeSessionService{
				session: tc.mockSession,
				err:     tc.mockSessionErr,
			}
			agentLoader := &fakeAgentLoader{
				agent: tc.mockAgent,
				err:   tc.mockAgentLoaderErr,
			}
			controller := NewRuntimeAPIController(sessionService, agentLoader, nil, 1*time.Second)

			body, _ := json.Marshal(tc.reqBody)
			req := httptest.NewRequest("POST", "/run", bytes.NewReader(body))
			rr := httptest.NewRecorder()

			err := controller.RunHandler(rr, req)

			if tc.wantErr != "" {
				if err == nil {
					t.Fatalf("expected error but got nil")
				}
				statusErr, ok := err.(statusError)
				if !ok {
					t.Fatalf("expected a statusError, got %T", err)
				}
				if !strings.Contains(statusErr.Err.Error(), tc.wantErr) {
					t.Errorf("handler returned wrong error message: got %q want %q", statusErr.Err.Error(), tc.wantErr)
				}
				if statusErr.Code != tc.wantStatusCode {
					t.Errorf("handler returned wrong status code: got %d want %d", statusErr.Code, tc.wantStatusCode)
				}
				return
			}
			if err != nil {
				t.Fatalf("handler returned unexpected error: %v", err)
			}
			if rr.Code != tc.wantStatusCode {
				t.Errorf("handler returned wrong status code: got %d want %d", rr.Code, tc.wantStatusCode)
			}

			if tc.wantRespBody != nil {
				var gotRespBody []models.Event
				if err := json.NewDecoder(rr.Body).Decode(&gotRespBody); err != nil {
					// An empty body is fine if we expect no events, but json.NewDecoder will error.
					if !(len(tc.wantRespBody) == 0 && rr.Body.String() == "") {
						t.Fatalf("failed to decode response body: %v, body: %s", err, rr.Body.String())
					}
				}
				if diff := cmp.Diff(tc.wantRespBody, gotRespBody); diff != "" {
					t.Errorf("handler returned unexpected body (-want +got):\n%s", diff)
				}
			}
		})
	}
}
