package ansiterm

import (
	"fmt"
	"strconv"
)

type TestAnsiEventHandler struct {
	FunctionCalls []string
}

func CreateTestAnsiEventHandler() *TestAnsiEventHandler {
	evtHandler := TestAnsiEventHandler{}
	evtHandler.FunctionCalls = make([]string, 0)
	return &evtHandler
}

func (h *TestAnsiEventHandler) recordCall(call string, params []string) {
	s := fmt.Sprintf("%s(%v)", call, params)
	h.FunctionCalls = append(h.FunctionCalls, s)
}

func (h *TestAnsiEventHandler) Print(b byte) error {
	h.recordCall("Print", []string{string(b)})
	return nil
}

func (h *TestAnsiEventHandler) Execute(b byte) error {
	h.recordCall("Execute", []string{string(b)})
	return nil
}

func (h *TestAnsiEventHandler) CUU(param int) error {
	h.recordCall("CUU", []string{strconv.Itoa(param)})
	return nil
}

func (h *TestAnsiEventHandler) CUD(param int) error {
	h.recordCall("CUD", []string{strconv.Itoa(param)})
	return nil
}

func (h *TestAnsiEventHandler) CUF(param int) error {
	h.recordCall("CUF", []string{strconv.Itoa(param)})
	return nil
}

func (h *TestAnsiEventHandler) CUB(param int) error {
	h.recordCall("CUB", []string{strconv.Itoa(param)})
	return nil
}

func (h *TestAnsiEventHandler) CNL(param int) error {
	h.recordCall("CNL", []string{strconv.Itoa(param)})
	return nil
}

func (h *TestAnsiEventHandler) CPL(param int) error {
	h.recordCall("CPL", []string{strconv.Itoa(param)})
	return nil
}

func (h *TestAnsiEventHandler) CHA(param int) error {
	h.recordCall("CHA", []string{strconv.Itoa(param)})
	return nil
}

func (h *TestAnsiEventHandler) VPA(param int) error {
	h.recordCall("VPA", []string{strconv.Itoa(param)})
	return nil
}

func (h *TestAnsiEventHandler) CUP(x int, y int) error {
	xS, yS := strconv.Itoa(x), strconv.Itoa(y)
	h.recordCall("CUP", []string{xS, yS})
	return nil
}

func (h *TestAnsiEventHandler) HVP(x int, y int) error {
	xS, yS := strconv.Itoa(x), strconv.Itoa(y)
	h.recordCall("HVP", []string{xS, yS})
	return nil
}

func (h *TestAnsiEventHandler) DECTCEM(visible bool) error {
	h.recordCall("DECTCEM", []string{strconv.FormatBool(visible)})
	return nil
}

func (h *TestAnsiEventHandler) DECOM(visible bool) error {
	h.recordCall("DECOM", []string{strconv.FormatBool(visible)})
	return nil
}

func (h *TestAnsiEventHandler) DECCOLM(use132 bool) error {
	h.recordCall("DECOLM", []string{strconv.FormatBool(use132)})
	return nil
}

func (h *TestAnsiEventHandler) ED(param int) error {
	h.recordCall("ED", []string{strconv.Itoa(param)})
	return nil
}

func (h *TestAnsiEventHandler) EL(param int) error {
	h.recordCall("EL", []string{strconv.Itoa(param)})
	return nil
}

func (h *TestAnsiEventHandler) IL(param int) error {
	h.recordCall("IL", []string{strconv.Itoa(param)})
	return nil
}

func (h *TestAnsiEventHandler) DL(param int) error {
	h.recordCall("DL", []string{strconv.Itoa(param)})
	return nil
}

func (h *TestAnsiEventHandler) ICH(param int) error {
	h.recordCall("ICH", []string{strconv.Itoa(param)})
	return nil
}

func (h *TestAnsiEventHandler) DCH(param int) error {
	h.recordCall("DCH", []string{strconv.Itoa(param)})
	return nil
}

func (h *TestAnsiEventHandler) SGR(params []int) error {
	strings := []string{}
	for _, v := range params {
		strings = append(strings, strconv.Itoa(v))
	}

	h.recordCall("SGR", strings)
	return nil
}

func (h *TestAnsiEventHandler) SU(param int) error {
	h.recordCall("SU", []string{strconv.Itoa(param)})
	return nil
}

func (h *TestAnsiEventHandler) SD(param int) error {
	h.recordCall("SD", []string{strconv.Itoa(param)})
	return nil
}

func (h *TestAnsiEventHandler) DA(params []string) error {
	h.recordCall("DA", params)
	return nil
}

func (h *TestAnsiEventHandler) DECSTBM(top int, bottom int) error {
	topS, bottomS := strconv.Itoa(top), strconv.Itoa(bottom)
	h.recordCall("DECSTBM", []string{topS, bottomS})
	return nil
}

func (h *TestAnsiEventHandler) RI() error {
	h.recordCall("RI", nil)
	return nil
}

func (h *TestAnsiEventHandler) IND() error {
	h.recordCall("IND", nil)
	return nil
}

func (h *TestAnsiEventHandler) Flush() error {
	return nil
}
