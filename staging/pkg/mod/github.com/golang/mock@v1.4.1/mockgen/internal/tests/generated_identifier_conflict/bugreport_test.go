package bugreport

import (
	"github.com/golang/mock/gomock"
	"testing"
)

func TestExample_Method(t *testing.T) {
	ctrl := gomock.NewController(t)
	m := NewMockExample(ctrl)
	m.EXPECT().Method(1, 2, 3, 4)

	m.Method(1, 2, 3, 4)

	ctrl.Finish()
}

func TestExample_VarargMethod(t *testing.T) {
	ctrl := gomock.NewController(t)
	m := NewMockExample(ctrl)
	m.EXPECT().VarargMethod(1, 2, 3, 4, 6, 7)

	m.VarargMethod(1, 2, 3, 4, 6, 7)

	ctrl.Finish()
}
