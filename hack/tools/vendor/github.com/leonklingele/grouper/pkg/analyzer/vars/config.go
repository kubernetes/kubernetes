package vars

type Config struct {
	RequireSingleVar bool // Require the use of a single global 'var' declaration only
	RequireGrouping  bool // Require the use of grouped global 'var' declarations
}
