package writer

type FakeGinkgoWriter struct {
	EventStream []string
}

func NewFake() *FakeGinkgoWriter {
	return &FakeGinkgoWriter{
		EventStream: []string{},
	}
}

func (writer *FakeGinkgoWriter) AddEvent(event string) {
	writer.EventStream = append(writer.EventStream, event)
}

func (writer *FakeGinkgoWriter) Truncate() {
	writer.EventStream = append(writer.EventStream, "TRUNCATE")
}

func (writer *FakeGinkgoWriter) DumpOut() {
	writer.EventStream = append(writer.EventStream, "DUMP")
}

func (writer *FakeGinkgoWriter) DumpOutWithHeader(header string) {
	writer.EventStream = append(writer.EventStream, "DUMP_WITH_HEADER: "+header)
}

func (writer *FakeGinkgoWriter) Bytes() []byte {
	writer.EventStream = append(writer.EventStream, "BYTES")
	return nil
}

func (writer *FakeGinkgoWriter) Write(data []byte) (n int, err error) {
	return 0, nil
}
