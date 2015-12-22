package graphite_test

import (
	"testing"
	"time"

	"github.com/BurntSushi/toml"
	"github.com/influxdb/influxdb/services/graphite"
)

func TestConfig_Parse(t *testing.T) {
	// Parse configuration.
	var c graphite.Config
	if _, err := toml.Decode(`
bind-address = ":8080"
database = "mydb"
enabled = true
protocol = "tcp"
batch-size=100
batch-pending=77
batch-timeout="1s"
consistency-level="one"
templates=["servers.* .host.measurement*"]
tags=["region=us-east"]
`, &c); err != nil {
		t.Fatal(err)
	}

	// Validate configuration.
	if c.BindAddress != ":8080" {
		t.Fatalf("unexpected bind address: %s", c.BindAddress)
	} else if c.Database != "mydb" {
		t.Fatalf("unexpected database selected: %s", c.Database)
	} else if c.Enabled != true {
		t.Fatalf("unexpected graphite enabled: %v", c.Enabled)
	} else if c.Protocol != "tcp" {
		t.Fatalf("unexpected graphite protocol: %s", c.Protocol)
	} else if c.BatchSize != 100 {
		t.Fatalf("unexpected graphite batch size: %d", c.BatchSize)
	} else if c.BatchPending != 77 {
		t.Fatalf("unexpected graphite batch pending: %d", c.BatchPending)
	} else if time.Duration(c.BatchTimeout) != time.Second {
		t.Fatalf("unexpected graphite batch timeout: %v", c.BatchTimeout)
	} else if c.ConsistencyLevel != "one" {
		t.Fatalf("unexpected graphite consistency setting: %s", c.ConsistencyLevel)
	}

	if len(c.Templates) != 1 && c.Templates[0] != "servers.* .host.measurement*" {
		t.Fatalf("unexpected graphite templates setting: %v", c.Templates)
	}
	if len(c.Tags) != 1 && c.Tags[0] != "regsion=us-east" {
		t.Fatalf("unexpected graphite templates setting: %v", c.Tags)
	}
}

func TestConfigValidateEmptyTemplate(t *testing.T) {
	c := &graphite.Config{}
	c.Templates = []string{""}
	if err := c.Validate(); err == nil {
		t.Errorf("config validate expected error. got nil")
	}

	c.Templates = []string{"     "}
	if err := c.Validate(); err == nil {
		t.Errorf("config validate expected error. got nil")
	}
}

func TestConfigValidateTooManyField(t *testing.T) {
	c := &graphite.Config{}
	c.Templates = []string{"a measurement b c"}
	if err := c.Validate(); err == nil {
		t.Errorf("config validate expected error. got nil")
	}
}

func TestConfigValidateTemplatePatterns(t *testing.T) {
	c := &graphite.Config{}
	c.Templates = []string{"*measurement"}
	if err := c.Validate(); err == nil {
		t.Errorf("config validate expected error. got nil")
	}

	c.Templates = []string{".host.region"}
	if err := c.Validate(); err == nil {
		t.Errorf("config validate expected error. got nil")
	}
}

func TestConfigValidateFilter(t *testing.T) {
	c := &graphite.Config{}
	c.Templates = []string{".server measurement*"}
	if err := c.Validate(); err == nil {
		t.Errorf("config validate expected error. got nil")
	}

	c.Templates = []string{".    .server measurement*"}
	if err := c.Validate(); err == nil {
		t.Errorf("config validate expected error. got nil")
	}

	c.Templates = []string{"server* measurement*"}
	if err := c.Validate(); err == nil {
		t.Errorf("config validate expected error. got nil")
	}
}

func TestConfigValidateTemplateTags(t *testing.T) {
	c := &graphite.Config{}
	c.Templates = []string{"*.server measurement* foo"}
	if err := c.Validate(); err == nil {
		t.Errorf("config validate expected error. got nil")
	}

	c.Templates = []string{"*.server measurement* foo=bar="}
	if err := c.Validate(); err == nil {
		t.Errorf("config validate expected error. got nil")
	}

	c.Templates = []string{"*.server measurement* foo=bar,"}
	if err := c.Validate(); err == nil {
		t.Errorf("config validate expected error. got nil")
	}

	c.Templates = []string{"*.server measurement* ="}
	if err := c.Validate(); err == nil {
		t.Errorf("config validate expected error. got nil")
	}
}

func TestConfigValidateDefaultTags(t *testing.T) {
	c := &graphite.Config{}
	c.Tags = []string{"foo"}
	if err := c.Validate(); err == nil {
		t.Errorf("config validate expected error. got nil")
	}

	c.Tags = []string{"foo=bar="}
	if err := c.Validate(); err == nil {
		t.Errorf("config validate expected error. got nil")
	}

	c.Tags = []string{"foo=bar", ""}
	if err := c.Validate(); err == nil {
		t.Errorf("config validate expected error. got nil")
	}

	c.Tags = []string{"="}
	if err := c.Validate(); err == nil {
		t.Errorf("config validate expected error. got nil")
	}
}

func TestConfigValidateFilterDuplicates(t *testing.T) {
	c := &graphite.Config{}
	c.Templates = []string{"foo measurement*", "foo .host.measurement"}
	if err := c.Validate(); err == nil {
		t.Errorf("config validate expected error. got nil")
	}

	// duplicate default templates
	c.Templates = []string{"measurement*", ".host.measurement"}
	if err := c.Validate(); err == nil {
		t.Errorf("config validate expected error. got nil")
	}

}
