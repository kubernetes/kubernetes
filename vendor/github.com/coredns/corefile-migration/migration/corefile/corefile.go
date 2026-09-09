package corefile

import (
	"strings"

	"github.com/coredns/caddy/caddyfile"
)

type Corefile struct {
	Servers []*Server
}

type Server struct {
	DomPorts []string
	Plugins  []*Plugin
}

type Plugin struct {
	Name    string
	Args    []string
	Options []*Option
}

type Option struct {
	Name string
	Args []string
}

func New(s string) (*Corefile, error) {
	c := Corefile{}
	cc := caddyfile.NewDispenser("migration", strings.NewReader(s))
	depth := 0
	var cSvr *Server
	var cPlg *Plugin
	for cc.Next() {
		if cc.Val() == "{" {
			depth += 1
			continue
		} else if cc.Val() == "}" {
			depth -= 1
			continue
		}
		val := cc.Val()
		args := cc.RemainingArgs()
		switch depth {
		case 0:
			c.Servers = append(c.Servers,
				&Server{
					DomPorts: append([]string{val}, args...),
				})
			cSvr = c.Servers[len(c.Servers)-1]
		case 1:
			cSvr.Plugins = append(cSvr.Plugins,
				&Plugin{
					Name: val,
					Args: args,
				})
			cPlg = cSvr.Plugins[len(cSvr.Plugins)-1]
		case 2:
			cPlg.Options = append(cPlg.Options,
				&Option{
					Name: val,
					Args: args,
				})
		}
	}
	return &c, nil
}

func (c *Corefile) ToString() (out string) {
	strs := []string{}
	for _, s := range c.Servers {
		strs = append(strs, s.ToString())
	}
	return strings.Join(strs, "\n")
}

func (s *Server) ToString() (out string) {
	str := strings.Join(escapeArgs(s.DomPorts), " ")
	strs := []string{}
	for _, p := range s.Plugins {
		strs = append(strs, strings.Repeat(" ", indent)+p.ToString())
	}
	if len(strs) > 0 {
		str += " {\n" + strings.Join(strs, "\n") + "\n}\n"
	}
	return str
}

func (p *Plugin) ToString() (out string) {
	str := strings.Join(append([]string{p.Name}, escapeArgs(p.Args)...), " ")
	strs := []string{}
	for _, o := range p.Options {
		strs = append(strs, strings.Repeat(" ", indent*2)+o.ToString())
	}
	if len(strs) > 0 {
		str += " {\n" + strings.Join(strs, "\n") + "\n" + strings.Repeat(" ", indent*1) + "}"
	}
	return str
}

func (o *Option) ToString() (out string) {
	str := strings.Join(append([]string{o.Name}, escapeArgs(o.Args)...), " ")
	return str
}

// escapeArgs returns the arguments list escaping and wrapping any argument containing whitespace in quotes
func escapeArgs(args []string) []string {
	var escapedArgs []string
	for _, a := range args {
		// if there is white space, wrap argument with quotes
		if len(strings.Fields(a)) > 1 {
			// escape quotes
			a = strings.Replace(a, "\"", "\\\"", -1)
			// wrap with quotes
			a = "\"" + a + "\""
		}
		escapedArgs = append(escapedArgs, a)
	}
	return escapedArgs
}

func (s *Server) FindMatch(def []*Server) (*Server, bool) {
NextServer:
	for _, sDef := range def {
		for i, dp := range sDef.DomPorts {
			if dp == "*" {
				continue
			}
			if dp == "***" {
				return sDef, true
			}
			if i >= len(s.DomPorts) || dp != s.DomPorts[i] {
				continue NextServer
			}
		}
		if len(sDef.DomPorts) != len(s.DomPorts) {
			continue
		}
		return sDef, true
	}
	return nil, false
}

func (p *Plugin) FindMatch(def []*Plugin) (*Plugin, bool) {
NextPlugin:
	for _, pDef := range def {
		if pDef.Name != p.Name {
			continue
		}
		for i, arg := range pDef.Args {
			if arg == "*" {
				continue
			}
			if arg == "***" {
				return pDef, true
			}
			if i >= len(p.Args) || arg != p.Args[i] {
				continue NextPlugin
			}
		}
		if len(pDef.Args) != len(p.Args) {
			continue
		}
		return pDef, true
	}
	return nil, false
}

func (o *Option) FindMatch(def []*Option) (*Option, bool) {
NextOption:
	for _, oDef := range def {
		if oDef.Name != o.Name {
			continue
		}
		for i, arg := range oDef.Args {
			if arg == "*" {
				continue
			}
			if arg == "***" {
				return oDef, true
			}
			if i >= len(o.Args) || arg != o.Args[i] {
				continue NextOption
			}
		}
		if len(oDef.Args) != len(o.Args) {
			continue
		}
		return oDef, true
	}
	return nil, false
}

const indent = 4
