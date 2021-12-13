package types

import (
	"regexp"
	"strconv"
	"strings"
)

func ParseFileFilters(filters []string) (FileFilters, error) {
	ffs := FileFilters{}
	for _, filter := range filters {
		ff := FileFilter{}
		if filter == "" {
			return nil, GinkgoErrors.InvalidFileFilter(filter)
		}
		components := strings.Split(filter, ":")
		if !(len(components) == 1 || len(components) == 2) {
			return nil, GinkgoErrors.InvalidFileFilter(filter)
		}

		var err error
		ff.Filename, err = regexp.Compile(components[0])
		if err != nil {
			return nil, err
		}
		if len(components) == 2 {
			lineFilters := strings.Split(components[1], ",")
			for _, lineFilter := range lineFilters {
				components := strings.Split(lineFilter, "-")
				if len(components) == 1 {
					line, err := strconv.Atoi(strings.TrimSpace(components[0]))
					if err != nil {
						return nil, GinkgoErrors.InvalidFileFilter(filter)
					}
					ff.LineFilters = append(ff.LineFilters, LineFilter{line, line + 1})
				} else if len(components) == 2 {
					line1, err := strconv.Atoi(strings.TrimSpace(components[0]))
					if err != nil {
						return nil, GinkgoErrors.InvalidFileFilter(filter)
					}
					line2, err := strconv.Atoi(strings.TrimSpace(components[1]))
					if err != nil {
						return nil, GinkgoErrors.InvalidFileFilter(filter)
					}
					ff.LineFilters = append(ff.LineFilters, LineFilter{line1, line2})
				} else {
					return nil, GinkgoErrors.InvalidFileFilter(filter)
				}
			}
		}
		ffs = append(ffs, ff)
	}
	return ffs, nil
}

type FileFilter struct {
	Filename    *regexp.Regexp
	LineFilters LineFilters
}

func (f FileFilter) Matches(locations []CodeLocation) bool {
	for _, location := range locations {
		if f.Filename.MatchString(location.FileName) &&
			f.LineFilters.Matches(location.LineNumber) {
			return true
		}

	}
	return false
}

type FileFilters []FileFilter

func (ffs FileFilters) Matches(locations []CodeLocation) bool {
	for _, ff := range ffs {
		if ff.Matches(locations) {
			return true
		}
	}

	return false
}

type LineFilter struct {
	Min int
	Max int
}

func (lf LineFilter) Matches(line int) bool {
	return lf.Min <= line && line < lf.Max
}

type LineFilters []LineFilter

func (lfs LineFilters) Matches(line int) bool {
	if len(lfs) == 0 {
		return true
	}

	for _, lf := range lfs {
		if lf.Matches(line) {
			return true
		}
	}
	return false
}
