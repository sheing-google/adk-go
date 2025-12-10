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

package adkrest

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/server/adkrest/internal/fakes"
)

func TestSecurityHeaders(t *testing.T) {
	// Setup a minimal config with a fake session service
	config := &launcher.Config{
		SessionService: &fakes.FakeSessionService{
			Sessions: make(map[fakes.SessionKey]fakes.TestSession),
		},
	}

	handler := NewHandler(config)

	req := httptest.NewRequest(http.MethodGet, "/apps/testApp/users/testUser/sessions", nil)
	rr := httptest.NewRecorder()

	handler.ServeHTTP(rr, req)

	expectedHeaders := map[string]string{
		"X-Content-Type-Options":  "nosniff",
		"X-Frame-Options":         "DENY",
		"Content-Security-Policy": "default-src 'self'",
		"Referrer-Policy":         "strict-origin-when-cross-origin",
	}

	for k, v := range expectedHeaders {
		if got := rr.Header().Get(k); got != v {
			t.Errorf("Header %q: got %q, want %q", k, got, v)
		}
	}
}
