// +build go1.8,codegen

package api

import (
	"testing"
)

func TestDocstring(t *testing.T) {
	cases := map[string]struct {
		In     string
		Expect string
	}{
		"non HTML": {
			In:     "Testing 1 2 3",
			Expect: "// Testing 1 2 3",
		},
		"link": {
			In:     `<a href="https://example.com">a link</a>`,
			Expect: "// a link (https://example.com)",
		},
		"link with space": {
			In:     `<a href=" https://example.com">a link</a>`,
			Expect: "// a link (https://example.com)",
		},
		"list HTML 01": {
			In:     "<ul><li>Testing 1 2 3</li> <li>FooBar</li></ul>",
			Expect: "//    * Testing 1 2 3\n// \n//    * FooBar",
		},
		"list HTML 02": {
			In:     "<ul> <li>Testing 1 2 3</li> <li>FooBar</li> </ul>",
			Expect: "//    * Testing 1 2 3\n// \n//    * FooBar",
		},
		"list HTML leading spaces": {
			In:     " <ul> <li>Testing 1 2 3</li> <li>FooBar</li> </ul>",
			Expect: "//    * Testing 1 2 3\n// \n//    * FooBar",
		},
		"list HTML paragraph": {
			In:     "<ul> <li> <p>Testing 1 2 3</p> </li><li> <p>FooBar</p></li></ul>",
			Expect: "//    * Testing 1 2 3\n// \n//    * FooBar",
		},
		"inline code HTML": {
			In:     "<ul> <li><code>Testing</code>: 1 2 3</li> <li>FooBar</li> </ul>",
			Expect: "//    * Testing: 1 2 3\n// \n//    * FooBar",
		},
		"complex list paragraph": {
			In:     "<ul> <li><p><code>FOO</code> Bar</p></li><li><p><code>Xyz</code> ABC</p></li></ul>",
			Expect: "//    * FOO Bar\n// \n//    * Xyz ABC",
		},
		"inline code in paragraph": {
			In:     "<p><code>Testing</code>: 1 2 3</p>",
			Expect: "// Testing: 1 2 3",
		},
		"root pre": {
			In:     "<pre><code>Testing</code></pre>",
			Expect: "//    Testing",
		},
		"paragraph": {
			In:     "<p>Testing 1 2 3</p>",
			Expect: "// Testing 1 2 3",
		},
		"wrap lines": {
			In:     "<span data-target-type=\"operation\" data-service=\"secretsmanager\" data-target=\"CreateSecret\">CreateSecret</span> <span data-target-type=\"structure\" data-service=\"secretsmanager\" data-target=\"SecretListEntry\">SecretListEntry</span> <span data-target-type=\"structure\" data-service=\"secretsmanager\" data-target=\"CreateSecret$SecretName\">SecretName</span> <span data-target-type=\"structure\" data-service=\"secretsmanager\" data-target=\"SecretListEntry$KmsKeyId\">KmsKeyId</span>",
			Expect: "// CreateSecret SecretListEntry SecretName KmsKeyId",
		},
		"links with spaces": {
			In:     "<p> Deletes the replication configuration from the bucket. For information about replication configuration, see <a href=\" https://docs.aws.amazon.com/AmazonS3/latest/dev/crr.html\">Cross-Region Replication (CRR)</a> in the <i>Amazon S3 Developer Guide</i>. </p>",
			Expect: "// Deletes the replication configuration from the bucket. For information about\n// replication configuration, see Cross-Region Replication (CRR) (https://docs.aws.amazon.com/AmazonS3/latest/dev/crr.html)\n// in the Amazon S3 Developer Guide.",
		},
	}

	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			t.Log("Input", c.In)
			actual := docstring(c.In)
			if e, a := c.Expect, actual; e != a {
				t.Errorf("expect %q, got %q", e, a)
			}
		})
	}
}
