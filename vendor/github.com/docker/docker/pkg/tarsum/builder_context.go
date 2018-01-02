package tarsum

// BuilderContext is an interface extending TarSum by adding the Remove method.
// In general there was concern about adding this method to TarSum itself
// so instead it is being added just to "BuilderContext" which will then
// only be used during the .dockerignore file processing
// - see builder/evaluator.go
type BuilderContext interface {
	TarSum
	Remove(string)
}

func (bc *tarSum) Remove(filename string) {
	for i, fis := range bc.sums {
		if fis.Name() == filename {
			bc.sums = append(bc.sums[:i], bc.sums[i+1:]...)
			// Note, we don't just return because there could be
			// more than one with this name
		}
	}
}
