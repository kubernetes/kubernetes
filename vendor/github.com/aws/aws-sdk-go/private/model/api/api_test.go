package api

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestStructNameWithFullName(t *testing.T) {
	a := API{
		Metadata: Metadata{
			ServiceFullName: "Amazon Service Name-100",
		},
	}
	assert.Equal(t, a.StructName(), "ServiceName100")
}

func TestStructNameWithAbbreviation(t *testing.T) {
	a := API{
		Metadata: Metadata{
			ServiceFullName:     "AWS Service Name-100",
			ServiceAbbreviation: "AWS SN100",
		},
	}
	assert.Equal(t, a.StructName(), "SN100")
}

func TestStructNameForExceptions(t *testing.T) {
	a := API{
		Metadata: Metadata{
			ServiceFullName: "Elastic Load Balancing",
		},
	}
	assert.Equal(t, a.StructName(), "ELB")

	a = API{
		Metadata: Metadata{
			ServiceFullName: "AWS Config",
		},
	}
	assert.Equal(t, a.StructName(), "ConfigService")
}
