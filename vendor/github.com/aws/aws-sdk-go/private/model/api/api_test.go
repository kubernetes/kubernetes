// +build 1.6,codegen

package api

import (
	"testing"
)

func TestStructNameWithFullName(t *testing.T) {
	a := API{
		Metadata: Metadata{
			ServiceFullName: "Amazon Service Name-100",
		},
	}
	if a.StructName() != "ServiceName100" {
		t.Errorf("API struct name should have been %s, but received %s", "ServiceName100", a.StructName())
	}
}

func TestStructNameWithAbbreviation(t *testing.T) {
	a := API{
		Metadata: Metadata{
			ServiceFullName:     "AWS Service Name-100",
			ServiceAbbreviation: "AWS SN100",
		},
	}
	if a.StructName() != "SN100" {
		t.Errorf("API struct name should have been %s, but received %s", "SN100", a.StructName())
	}
}

func TestStructNameForExceptions(t *testing.T) {
	serviceAliases = map[string]string{}
	serviceAliases["elasticloadbalancing"] = "ELB"
	serviceAliases["config"] = "ConfigService"

	a := API{
		Metadata: Metadata{
			ServiceFullName: "Elastic Load Balancing",
		},
	}
	if a.StructName() != "ELB" {
		t.Errorf("API struct name should have been %s, but received %s", "ELB", a.StructName())
	}

	a = API{
		Metadata: Metadata{
			ServiceFullName: "AWS Config",
		},
	}
	if a.StructName() != "ConfigService" {
		t.Errorf("API struct name should have been %s, but received %s", "ConfigService", a.StructName())
	}
}
