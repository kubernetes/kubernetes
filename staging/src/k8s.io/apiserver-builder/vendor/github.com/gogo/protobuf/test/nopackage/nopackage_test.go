package nopackage

import (
	"testing"
)

func TestNoPackage(t *testing.T) {
	//should compile
	_ = (&M{}).Marshal
}
