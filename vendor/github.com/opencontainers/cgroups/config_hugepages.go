package cgroups

type HugepageLimit struct {
	// which type of hugepage to limit.
	Pagesize string `json:"page_size"`

	// usage limit for hugepage.
	Limit uint64 `json:"limit"`
}
