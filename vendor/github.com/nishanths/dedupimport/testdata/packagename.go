package pkg

import (
	"github.com/foo/shellutil"
	"gopkg.in/go-git.v3"
	"google.io/meta/v1"
	"github.com/nishanths/go-hgconfig"
	"github.com/nishanths/lyft-go"
	
	a "github.com/foo/shellutil"
	b "gopkg.in/go-git.v3"
	c "google.io/meta/v1"
	d "github.com/nishanths/go-hgconfig"
	e "github.com/nishanths/lyft-go"
)

func foo() {
	_ = a.x
	_ = b.x
	{ 
		_ = c.x 
	}
	_ = d.x
	{ 
		_ = e.x
	}
}