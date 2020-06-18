//events_test.go
package events

import (
	"net/http"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/resource"
	"k8s.io/client-go/rest/fake"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
	"k8s.io/kubectl/pkg/scheme"
)

func TestGetSchemaObject(t *testing.T) {
	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	codec := scheme.Codecs.LegacyCodec(corev1.SchemeGroupVersion)
	t.Logf("%v", string(runtime.EncodeOrDie(codec, &corev1.ReplicationController{ObjectMeta: metav1.ObjectMeta{Name: "foo"}})))

	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Resp:                 &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, &corev1.Event{ObjectMeta: metav1.ObjectMeta{Name: "foo"}})},
	}
	tf.ClientConfigVal = cmdtesting.DefaultClientConfig()

	streams, _, buf, _ := genericclioptions.NewTestIOStreams()
	cmd := NewCmdEvents(tf, streams)
	cmd.Run(cmd, []string{"events"})

	if !strings.Contains(buf.String(), "foo") {
		t.Errorf("unexpected output: %s", buf.String())
	}
}
