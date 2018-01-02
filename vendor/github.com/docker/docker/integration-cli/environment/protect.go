package environment

// ProtectImage adds the specified image(s) to be protected in case of clean
func (e *Execution) ProtectImage(t testingT, images ...string) {
	for _, image := range images {
		e.protectedElements.images[image] = struct{}{}
	}
}

type protectedElements struct {
	images map[string]struct{}
}
