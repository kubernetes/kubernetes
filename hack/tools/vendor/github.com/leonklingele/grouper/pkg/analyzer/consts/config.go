package consts

type Config struct {
	RequireSingleConst bool // Require the use of a single global 'const' declaration only
	RequireGrouping    bool // Require the use of grouped global 'const' declarations
}
