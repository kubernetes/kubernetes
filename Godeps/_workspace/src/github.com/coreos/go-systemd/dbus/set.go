package dbus

type set struct {
	data map[string]bool
}

func (s *set) Add(value string) {
	s.data[value] = true
}

func (s *set) Remove(value string) {
	delete(s.data, value)
}

func (s *set) Contains(value string) (exists bool) {
	_, exists = s.data[value]
	return
}

func (s *set) Length() (int) {
	return len(s.data)
}

func (s *set) Values() (values []string) {
	 for val, _ := range s.data {
		values = append(values, val)
	 }
	 return
}

func newSet() (*set) {
	return &set{make(map[string] bool)}
}
