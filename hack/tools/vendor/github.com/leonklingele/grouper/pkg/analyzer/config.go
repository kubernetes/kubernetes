package analyzer

import (
	"github.com/leonklingele/grouper/pkg/analyzer/consts"
	"github.com/leonklingele/grouper/pkg/analyzer/imports"
	"github.com/leonklingele/grouper/pkg/analyzer/types"
	"github.com/leonklingele/grouper/pkg/analyzer/vars"
)

type Config struct {
	ConstsConfig  *consts.Config
	ImportsConfig *imports.Config
	TypesConfig   *types.Config
	VarsConfig    *vars.Config
}
