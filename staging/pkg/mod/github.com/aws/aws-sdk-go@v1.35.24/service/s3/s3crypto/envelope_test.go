// +build go1.7

package s3crypto

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestEnvelope_UnmarshalJSON(t *testing.T) {
	cases := map[string]struct {
		content  []byte
		expected Envelope
		actual   Envelope
	}{
		"string json numbers": {
			content: []byte(`{
  "x-amz-iv": "iv",
  "x-amz-key-v2": "key",
  "x-amz-matdesc": "{\"aws:x-amz-cek-alg\":\"AES/GCM/NoPadding\"}",
  "x-amz-wrap-alg": "kms+context",
  "x-amz-cek-alg": "AES/GCM/NoPadding",
  "x-amz-tag-len": "128",
  "x-amz-unencrypted-content-length": "1024"
}
`),
			expected: Envelope{
				IV:                    "iv",
				CipherKey:             "key",
				MatDesc:               `{"aws:x-amz-cek-alg":"AES/GCM/NoPadding"}`,
				WrapAlg:               "kms+context",
				CEKAlg:                "AES/GCM/NoPadding",
				TagLen:                "128",
				UnencryptedContentLen: "1024",
			},
		},
		"integer json numbers": {
			content: []byte(`{
  "x-amz-iv": "iv",
  "x-amz-key-v2": "key",
  "x-amz-matdesc": "{\"aws:x-amz-cek-alg\":\"AES/GCM/NoPadding\"}",
  "x-amz-wrap-alg": "kms+context",
  "x-amz-cek-alg": "AES/GCM/NoPadding",
  "x-amz-tag-len": 128,
  "x-amz-unencrypted-content-length": 1024
}
`),
			expected: Envelope{
				IV:                    "iv",
				CipherKey:             "key",
				MatDesc:               `{"aws:x-amz-cek-alg":"AES/GCM/NoPadding"}`,
				WrapAlg:               "kms+context",
				CEKAlg:                "AES/GCM/NoPadding",
				TagLen:                "128",
				UnencryptedContentLen: "1024",
			},
		},
		"null json numbers": {
			content: []byte(`{
  "x-amz-iv": "iv",
  "x-amz-key-v2": "key",
  "x-amz-matdesc": "{\"aws:x-amz-cek-alg\":\"AES/GCM/NoPadding\"}",
  "x-amz-wrap-alg": "kms+context",
  "x-amz-cek-alg": "AES/GCM/NoPadding",
  "x-amz-tag-len": null,
  "x-amz-unencrypted-content-length": null
}
`),
			expected: Envelope{
				IV:        "iv",
				CipherKey: "key",
				MatDesc:   `{"aws:x-amz-cek-alg":"AES/GCM/NoPadding"}`,
				WrapAlg:   "kms+context",
				CEKAlg:    "AES/GCM/NoPadding",
			},
		},
		"no json numbers": {
			content: []byte(`{
  "x-amz-iv": "iv",
  "x-amz-key-v2": "key",
  "x-amz-matdesc": "{\"aws:x-amz-cek-alg\":\"AES/GCM/NoPadding\"}",
  "x-amz-wrap-alg": "kms+context",
  "x-amz-cek-alg": "AES/GCM/NoPadding"
}
`),
			expected: Envelope{
				IV:        "iv",
				CipherKey: "key",
				MatDesc:   `{"aws:x-amz-cek-alg":"AES/GCM/NoPadding"}`,
				WrapAlg:   "kms+context",
				CEKAlg:    "AES/GCM/NoPadding",
			},
		},
	}

	for name, tt := range cases {
		t.Run(name, func(t *testing.T) {
			err := json.Unmarshal(tt.content, &tt.actual)
			if err != nil {
				t.Errorf("expected no error, got %v", err)
			}
			if !reflect.DeepEqual(tt.expected, tt.actual) {
				t.Errorf("expected %v, got %v", tt.expected, tt.actual)
			}
		})
	}
}
