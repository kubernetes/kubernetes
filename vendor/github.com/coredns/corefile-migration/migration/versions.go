package migration

import (
	"errors"
	"github.com/coredns/corefile-migration/migration/corefile"
)

type plugin struct {
	status     string
	replacedBy string
	additional string
	options    map[string]option
	action     pluginActionFn // action affecting this plugin only
	add        serverActionFn // action to add a new plugin to the server block
	downAction pluginActionFn // downgrade action affecting this plugin only
}

type option struct {
	status     string
	replacedBy string
	additional string
	action     optionActionFn // action affecting this option only
	add        pluginActionFn // action to add the option to the plugin
	downAction optionActionFn // downgrade action affecting this option only
}

type release struct {
	k8sReleases    []string
	nextVersion    string
	priorVersion   string
	dockerImageSHA string
	plugins        map[string]plugin // list of plugins with deprecation status and migration actions

	// postProcess is a post processing action to take on the corefile as a whole.  Used for complex migration
	//   tasks that dont fit well into the modular plugin/option migration framework. For example, when the
	//   action on a plugin would need to extend beyond the scope of that plugin (affecting other plugins, or
	//   server blocks, etc). e.g. Splitting plugins out into separate server blocks.
	postProcess corefileAction

	// defaultConf holds the default Corefile template packaged with the corresponding k8sReleases.
	// Wildcards are used for fuzzy matching:
	//   "*"   matches exactly one token
	//   "***" matches 0 all remaining tokens on the line
	// Order of server blocks, plugins, and options does not matter.
	// Order of arguments does matter.
	defaultConf string
}

type corefileAction func(*corefile.Corefile) (*corefile.Corefile, error)
type serverActionFn func(*corefile.Server) (*corefile.Server, error)
type pluginActionFn func(*corefile.Plugin) (*corefile.Plugin, error)
type optionActionFn func(*corefile.Option) (*corefile.Option, error)

func removePlugin(*corefile.Plugin) (*corefile.Plugin, error) { return nil, nil }
func removeOption(*corefile.Option) (*corefile.Option, error) { return nil, nil }

func renamePlugin(p *corefile.Plugin, to string) (*corefile.Plugin, error) {
	p.Name = to
	return p, nil
}

func addToServerBlockWithPlugins(sb *corefile.Server, newPlugin *corefile.Plugin, with []string) (*corefile.Server, error) {
	if len(with) == 0 {
		// add to all blocks
		sb.Plugins = append(sb.Plugins, newPlugin)
		return sb, nil
	}
	for _, p := range sb.Plugins {
		for _, w := range with {
			if w == p.Name {
				// add to this block
				sb.Plugins = append(sb.Plugins, newPlugin)
				return sb, nil
			}
		}
	}
	return sb, nil
}

func addToKubernetesServerBlocks(sb *corefile.Server, newPlugin *corefile.Plugin) (*corefile.Server, error) {
	return addToServerBlockWithPlugins(sb, newPlugin, []string{"kubernetes"})
}

func addToForwardingServerBlocks(sb *corefile.Server, newPlugin *corefile.Plugin) (*corefile.Server, error) {
	return addToServerBlockWithPlugins(sb, newPlugin, []string{"forward", "proxy"})
}

func addToAllServerBlocks(sb *corefile.Server, newPlugin *corefile.Plugin) (*corefile.Server, error) {
	return addToServerBlockWithPlugins(sb, newPlugin, []string{})
}

func addOptionToPlugin(pl *corefile.Plugin, newOption *corefile.Option) (*corefile.Plugin, error) {
	pl.Options = append(pl.Options, newOption)
	return pl, nil
}

var Versions = map[string]release{
	"1.6.5": {
		priorVersion:   "1.6.4",
		dockerImageSHA: "7ec975f167d815311a7136c32e70735f0d00b73781365df1befd46ed35bd4fe7",
		defaultConf: `.:53 {
    errors
    health {
        lameduck 5s
    }
    kubernetes * *** {
        pods insecure
        upstream
        fallthrough in-addr.arpa ip6.arpa
        ttl 30
    }
    prometheus :9153
    forward . *
    cache 30
    loop
    reload
    loadbalance
}`,
		plugins: map[string]plugin{
			"errors": {
				options: map[string]option{
					"consolidate": {},
				},
			},
			"log": {
				options: map[string]option{
					"class": {},
				},
			},
			"health": {
				options: map[string]option{
					"lameduck": {
						status: newdefault,
						add: func(c *corefile.Plugin) (*corefile.Plugin, error) {
							return addOptionToPlugin(c, &corefile.Option{Name: "lameduck 5s"})
						},
						downAction: removeOption,
					},
				},
			},
			"ready":    {},
			"autopath": {},
			"kubernetes": {
				options: map[string]option{
					"endpoint": {
						status: ignored,
						action: useFirstArgumentOnly,
					},
					"tls":                {},
					"kubeconfig":         {},
					"namespaces":         {},
					"labels":             {},
					"pods":               {},
					"endpoint_pod_names": {},
					"upstream": {
						status: ignored,
						action: removeOption,
					},
					"ttl":         {},
					"noendpoints": {},
					"transfer":    {},
					"fallthrough": {},
					"ignore":      {},
				},
			},
			"k8s_external": {
				options: map[string]option{
					"apex": {},
					"ttl":  {},
				},
			},
			"prometheus": {},
			"forward": {
				options: map[string]option{
					"except":         {},
					"force_tcp":      {},
					"prefer_udp":     {},
					"expire":         {},
					"max_fails":      {},
					"tls":            {},
					"tls_servername": {},
					"policy":         {},
					"health_check":   {},
				},
			},
			"cache": {
				options: map[string]option{
					"success":  {},
					"denial":   {},
					"prefetch": {},
				},
			},
			"loop":        {},
			"reload":      {},
			"loadbalance": {},
		},
	},
	"1.6.4": {
		nextVersion:    "1.6.5",
		priorVersion:   "1.6.3",
		dockerImageSHA: "493ee88e1a92abebac67cbd4b5658b4730e0f33512461442d8d9214ea6734a9b",
		plugins: map[string]plugin{
			"errors": {
				options: map[string]option{
					"consolidate": {},
				},
			},
			"log": {
				options: map[string]option{
					"class": {},
				},
			},
			"health":   {},
			"ready":    {},
			"autopath": {},
			"kubernetes": {
				options: map[string]option{
					"endpoint": {
						status: ignored,
						action: useFirstArgumentOnly,
					},
					"tls":                {},
					"kubeconfig":         {},
					"namespaces":         {},
					"labels":             {},
					"pods":               {},
					"endpoint_pod_names": {},
					"upstream": {
						status: ignored,
						action: removeOption,
					},
					"ttl":         {},
					"noendpoints": {},
					"transfer":    {},
					"fallthrough": {},
					"ignore":      {},
				},
			},
			"k8s_external": {
				options: map[string]option{
					"apex": {},
					"ttl":  {},
				},
			},
			"prometheus": {},
			"forward": {
				options: map[string]option{
					"except":         {},
					"force_tcp":      {},
					"prefer_udp":     {},
					"expire":         {},
					"max_fails":      {},
					"tls":            {},
					"tls_servername": {},
					"policy":         {},
					"health_check":   {},
				},
			},
			"cache": {
				options: map[string]option{
					"success":  {},
					"denial":   {},
					"prefetch": {},
				},
			},
			"loop":        {},
			"reload":      {},
			"loadbalance": {},
		},
	},
	"1.6.3": {
		nextVersion:    "1.6.4",
		priorVersion:   "1.6.2",
		dockerImageSHA: "cfa7236dab4e3860881fdf755880ff8361e42f6cba2e3775ae48e2d46d22f7ba",
		plugins: map[string]plugin{
			"errors": {
				options: map[string]option{
					"consolidate": {},
				},
			},
			"log": {
				options: map[string]option{
					"class": {},
				},
			},
			"health":   {},
			"ready":    {},
			"autopath": {},
			"kubernetes": {
				options: map[string]option{
					"endpoint": {
						status: ignored,
						action: useFirstArgumentOnly,
					},
					"tls":                {},
					"kubeconfig":         {},
					"namespaces":         {},
					"labels":             {},
					"pods":               {},
					"endpoint_pod_names": {},
					"upstream": {
						status: ignored,
						action: removeOption,
					},
					"ttl":         {},
					"noendpoints": {},
					"transfer":    {},
					"fallthrough": {},
					"ignore":      {},
				},
			},
			"k8s_external": {
				options: map[string]option{
					"apex": {},
					"ttl":  {},
				},
			},
			"prometheus": {},
			"forward": {
				options: map[string]option{
					"except":         {},
					"force_tcp":      {},
					"prefer_udp":     {},
					"expire":         {},
					"max_fails":      {},
					"tls":            {},
					"tls_servername": {},
					"policy":         {},
					"health_check":   {},
				},
			},
			"cache": {
				options: map[string]option{
					"success":  {},
					"denial":   {},
					"prefetch": {},
				},
			},
			"loop":        {},
			"reload":      {},
			"loadbalance": {},
		},
	},
	"1.6.2": {
		nextVersion:    "1.6.3",
		priorVersion:   "1.6.1",
		dockerImageSHA: "12eb885b8685b1b13a04ecf5c23bc809c2e57917252fd7b0be9e9c00644e8ee5",
		defaultConf: `.:53 {
    errors
    health
    kubernetes * *** {
        pods insecure
        upstream
        fallthrough in-addr.arpa ip6.arpa
        ttl 30
    }
    prometheus :9153
    forward . *
    cache 30
    loop
    reload
    loadbalance
}`,
		plugins: map[string]plugin{
			"errors": {
				options: map[string]option{
					"consolidate": {},
				},
			},
			"log": {
				options: map[string]option{
					"class": {},
				},
			},
			"health":   {},
			"ready":    {},
			"autopath": {},
			"kubernetes": {
				options: map[string]option{
					"endpoint": {
						status: ignored,
						action: useFirstArgumentOnly,
					},
					"tls":                {},
					"kubeconfig":         {},
					"namespaces":         {},
					"labels":             {},
					"pods":               {},
					"endpoint_pod_names": {},
					"upstream": {
						status: ignored,
						action: removeOption,
					},
					"ttl":         {},
					"noendpoints": {},
					"transfer":    {},
					"fallthrough": {},
					"ignore":      {},
				},
			},
			"k8s_external": {
				options: map[string]option{
					"apex": {},
					"ttl":  {},
				},
			},
			"prometheus": {},
			"forward": {
				options: map[string]option{
					"except":         {},
					"force_tcp":      {},
					"prefer_udp":     {},
					"expire":         {},
					"max_fails":      {},
					"tls":            {},
					"tls_servername": {},
					"policy":         {},
					"health_check":   {},
				},
			},
			"cache": {
				options: map[string]option{
					"success":  {},
					"denial":   {},
					"prefetch": {},
				},
			},
			"loop":        {},
			"reload":      {},
			"loadbalance": {},
		},
	},
	"1.6.1": {
		nextVersion:    "1.6.2",
		priorVersion:   "1.6.0",
		dockerImageSHA: "9ae3b6fcac4ee821362277de6bd8fd2236fa7d3e19af2ef0406d80b595620a7a",
		plugins: map[string]plugin{
			"errors": {
				options: map[string]option{
					"consolidate": {},
				},
			},
			"log": {
				options: map[string]option{
					"class": {},
				},
			},
			"health":   {},
			"ready":    {},
			"autopath": {},
			"kubernetes": {
				options: map[string]option{
					"endpoint": {
						status: ignored,
						action: useFirstArgumentOnly,
					},
					"tls":                {},
					"kubeconfig":         {},
					"namespaces":         {},
					"labels":             {},
					"pods":               {},
					"endpoint_pod_names": {},
					"upstream": {
						status: ignored,
						action: removeOption,
					},
					"ttl":         {},
					"noendpoints": {},
					"transfer":    {},
					"fallthrough": {},
					"ignore":      {},
				},
			},
			"k8s_external": {
				options: map[string]option{
					"apex": {},
					"ttl":  {},
				},
			},
			"prometheus": {},
			"forward": {
				options: map[string]option{
					"except":         {},
					"force_tcp":      {},
					"prefer_udp":     {},
					"expire":         {},
					"max_fails":      {},
					"tls":            {},
					"tls_servername": {},
					"policy":         {},
					"health_check":   {},
				},
			},
			"cache": {
				options: map[string]option{
					"success":  {},
					"denial":   {},
					"prefetch": {},
				},
			},
			"loop":        {},
			"reload":      {},
			"loadbalance": {},
		},
	},
	"1.6.0": {
		nextVersion:    "1.6.1",
		priorVersion:   "1.5.2",
		dockerImageSHA: "263d03f2b889a75a0b91e035c2a14d45d7c1559c53444c5f7abf3a76014b779d",
		plugins: map[string]plugin{
			"errors": {
				options: map[string]option{
					"consolidate": {},
				},
			},
			"log": {
				options: map[string]option{
					"class": {},
				},
			},
			"health":   {},
			"ready":    {},
			"autopath": {},
			"kubernetes": {
				options: map[string]option{
					"resyncperiod": {
						status: removed,
						action: removeOption,
					},
					"endpoint": {
						status: ignored,
						action: useFirstArgumentOnly,
					},
					"tls":                {},
					"kubeconfig":         {},
					"namespaces":         {},
					"labels":             {},
					"pods":               {},
					"endpoint_pod_names": {},
					"upstream": {
						status: ignored,
						action: removeOption,
					},
					"ttl":         {},
					"noendpoints": {},
					"transfer":    {},
					"fallthrough": {},
					"ignore":      {},
				},
			},
			"k8s_external": {
				options: map[string]option{
					"apex": {},
					"ttl":  {},
				},
			},
			"prometheus": {},
			"forward": {
				options: map[string]option{
					"except":         {},
					"force_tcp":      {},
					"prefer_udp":     {},
					"expire":         {},
					"max_fails":      {},
					"tls":            {},
					"tls_servername": {},
					"policy":         {},
					"health_check":   {},
				},
			},
			"cache": {
				options: map[string]option{
					"success":  {},
					"denial":   {},
					"prefetch": {},
				},
			},
			"loop":        {},
			"reload":      {},
			"loadbalance": {},
		},
	},
	"1.5.2": {
		nextVersion:    "1.6.0",
		priorVersion:   "1.5.1",
		dockerImageSHA: "586d15ec14911ee680ac9c5af20ff24b9d1412fbbf0e05862ee1f5c37baa65b2",
		plugins: map[string]plugin{
			"errors": {
				options: map[string]option{
					"consolidate": {},
				},
			},
			"log": {
				options: map[string]option{
					"class": {},
				},
			},
			"health":   {},
			"ready":    {},
			"autopath": {},
			"kubernetes": {
				options: map[string]option{
					"resyncperiod": {
						status: deprecated,
						action: removeOption,
					},
					"endpoint": {
						status: ignored,
						action: useFirstArgumentOnly,
					},
					"tls":                {},
					"kubeconfig":         {},
					"namespaces":         {},
					"labels":             {},
					"pods":               {},
					"endpoint_pod_names": {},
					"upstream": {
						status: ignored,
						action: removeOption,
					},
					"ttl":         {},
					"noendpoints": {},
					"transfer":    {},
					"fallthrough": {},
					"ignore":      {},
				},
			},
			"k8s_external": {
				options: map[string]option{
					"apex": {},
					"ttl":  {},
				},
			},
			"prometheus": {},
			"forward": {
				options: map[string]option{
					"except":         {},
					"force_tcp":      {},
					"prefer_udp":     {},
					"expire":         {},
					"max_fails":      {},
					"tls":            {},
					"tls_servername": {},
					"policy":         {},
					"health_check":   {},
				},
			},
			"cache": {
				options: map[string]option{
					"success":  {},
					"denial":   {},
					"prefetch": {},
				},
			},
			"loop":        {},
			"reload":      {},
			"loadbalance": {},
		},
	},
	"1.5.1": {
		nextVersion:    "1.5.2",
		priorVersion:   "1.5.0",
		dockerImageSHA: "451817637035535ae1fc8639753b453fa4b781d0dea557d5da5cb3c131e62ef5",
		plugins: map[string]plugin{
			"errors": {
				options: map[string]option{
					"consolidate": {},
				},
			},
			"log": {
				options: map[string]option{
					"class": {},
				},
			},
			"health":   {},
			"ready":    {},
			"autopath": {},
			"kubernetes": {
				options: map[string]option{
					"resyncperiod": {
						status: deprecated,
						action: removeOption,
					},
					"endpoint": {
						status: ignored,
						action: useFirstArgumentOnly,
					},
					"tls":                {},
					"kubeconfig":         {},
					"namespaces":         {},
					"labels":             {},
					"pods":               {},
					"endpoint_pod_names": {},
					"upstream": {
						status: ignored,
						action: removeOption,
					},
					"ttl":         {},
					"noendpoints": {},
					"transfer":    {},
					"fallthrough": {},
					"ignore":      {},
				},
			},
			"k8s_external": {
				options: map[string]option{
					"apex": {},
					"ttl":  {},
				},
			},
			"prometheus": {},
			"forward": {
				options: map[string]option{
					"except":         {},
					"force_tcp":      {},
					"prefer_udp":     {},
					"expire":         {},
					"max_fails":      {},
					"tls":            {},
					"tls_servername": {},
					"policy":         {},
					"health_check":   {},
				},
			},
			"cache": {
				options: map[string]option{
					"success":  {},
					"denial":   {},
					"prefetch": {},
				},
			},
			"loop":        {},
			"reload":      {},
			"loadbalance": {},
		},
	},
	"1.5.0": {
		nextVersion:    "1.5.1",
		priorVersion:   "1.4.0",
		dockerImageSHA: "e83beb5e43f8513fa735e77ffc5859640baea30a882a11cc75c4c3244a737d3c",
		plugins: map[string]plugin{
			"errors": {
				options: map[string]option{
					"consolidate": {},
				},
			},
			"log": {
				options: map[string]option{
					"class": {},
				},
			},
			"health": {},
			"ready": {
				status: newdefault,
				add: func(c *corefile.Server) (*corefile.Server, error) {
					return addToKubernetesServerBlocks(c, &corefile.Plugin{Name: "ready"})
				},
				downAction: removePlugin,
			},
			"autopath": {},
			"kubernetes": {
				options: map[string]option{
					"resyncperiod": {
						status: deprecated,
						action: removeOption,
					},
					"endpoint": {
						status: ignored,
						action: useFirstArgumentOnly,
					},
					"tls":                {},
					"kubeconfig":         {},
					"namespaces":         {},
					"labels":             {},
					"pods":               {},
					"endpoint_pod_names": {},
					"upstream": {
						status: ignored,
						action: removeOption,
					},
					"ttl":         {},
					"noendpoints": {},
					"transfer":    {},
					"fallthrough": {},
					"ignore":      {},
				},
			},
			"k8s_external": {
				options: map[string]option{
					"apex": {},
					"ttl":  {},
				},
			},
			"prometheus": {},
			"proxy": {
				status:     removed,
				replacedBy: "forward",
				action:     proxyToForwardPluginAction,
				options:    proxyToForwardOptionsMigrations,
			},
			"forward": {
				options: map[string]option{
					"except":         {},
					"force_tcp":      {},
					"prefer_udp":     {},
					"expire":         {},
					"max_fails":      {},
					"tls":            {},
					"tls_servername": {},
					"policy":         {},
					"health_check":   {},
				},
			},
			"cache": {
				options: map[string]option{
					"success":  {},
					"denial":   {},
					"prefetch": {},
				},
			},
			"loop":        {},
			"reload":      {},
			"loadbalance": {},
		},
		postProcess: breakForwardStubDomainsIntoServerBlocks,
	},
	"1.4.0": {
		nextVersion:    "1.5.0",
		priorVersion:   "1.3.1",
		dockerImageSHA: "70a92e9f6fc604f9b629ca331b6135287244a86612f550941193ec7e12759417",
		plugins: map[string]plugin{
			"errors": {
				options: map[string]option{
					"consolidate": {},
				},
			},
			"log": {
				options: map[string]option{
					"class": {},
				},
			},
			"health":   {},
			"autopath": {},
			"kubernetes": {
				options: map[string]option{
					"resyncperiod": {},
					"endpoint": {
						status: ignored,
						action: useFirstArgumentOnly,
					},
					"tls":                {},
					"kubeconfig":         {},
					"namespaces":         {},
					"labels":             {},
					"pods":               {},
					"endpoint_pod_names": {},
					"upstream": {
						status: deprecated,
						action: removeOption,
					},
					"ttl":         {},
					"noendpoints": {},
					"transfer":    {},
					"fallthrough": {},
					"ignore":      {},
				},
			},
			"k8s_external": {
				options: map[string]option{
					"apex": {},
					"ttl":  {},
				},
			},
			"prometheus": {},
			"proxy": {
				status:     deprecated,
				replacedBy: "forward",
				action:     proxyToForwardPluginAction,
				options:    proxyToForwardOptionsMigrations,
			},
			"forward": {
				options: map[string]option{
					"except":         {},
					"force_tcp":      {},
					"prefer_udp":     {},
					"expire":         {},
					"max_fails":      {},
					"tls":            {},
					"tls_servername": {},
					"policy":         {},
					"health_check":   {},
				},
			},
			"cache": {
				options: map[string]option{
					"success":  {},
					"denial":   {},
					"prefetch": {},
				},
			},
			"loop":        {},
			"reload":      {},
			"loadbalance": {},
		},
		postProcess: breakForwardStubDomainsIntoServerBlocks,
	},
	"1.3.1": {
		nextVersion:    "1.4.0",
		priorVersion:   "1.3.0",
		k8sReleases:    []string{"1.15", "1.14"},
		dockerImageSHA: "02382353821b12c21b062c59184e227e001079bb13ebd01f9d3270ba0fcbf1e4",
		defaultConf: `.:53 {
    errors
    health
    kubernetes * *** {
        pods insecure
        upstream
        fallthrough in-addr.arpa ip6.arpa
        ttl 30
    }
    prometheus :9153
    forward . *
    cache 30
    loop
    reload
    loadbalance
}`,
		plugins: map[string]plugin{
			"errors": {
				options: map[string]option{
					"consolidate": {},
				},
			},
			"log": {
				options: map[string]option{
					"class": {},
				},
			},
			"health":   {},
			"autopath": {},
			"kubernetes": {
				options: map[string]option{
					"resyncperiod": {},
					"endpoint": {
						status: deprecated,
						action: useFirstArgumentOnly,
					},
					"tls":                {},
					"kubeconfig":         {},
					"namespaces":         {},
					"labels":             {},
					"pods":               {},
					"endpoint_pod_names": {},
					"upstream":           {},
					"ttl":                {},
					"noendpoints":        {},
					"transfer":           {},
					"fallthrough":        {},
					"ignore":             {},
				},
			},
			"k8s_external": {
				options: map[string]option{
					"apex": {},
					"ttl":  {},
				},
			},
			"prometheus": {},
			"proxy": {
				options: map[string]option{
					"policy":       {},
					"fail_timeout": {},
					"max_fails":    {},
					"health_check": {},
					"except":       {},
					"spray":        {},
					"protocol":     {},
				},
			},
			"forward": {
				options: map[string]option{
					"except":         {},
					"force_tcp":      {},
					"prefer_udp":     {},
					"expire":         {},
					"max_fails":      {},
					"tls":            {},
					"tls_servername": {},
					"policy":         {},
					"health_check":   {},
				},
			},
			"cache": {
				options: map[string]option{
					"success":  {},
					"denial":   {},
					"prefetch": {},
				},
			},
			"loop":        {},
			"reload":      {},
			"loadbalance": {},
		},
	},
	"1.3.0": {
		nextVersion:    "1.3.1",
		priorVersion:   "1.2.6",
		dockerImageSHA: "e030773c7fee285435ed7fc7623532ee54c4c1c4911fb24d95cd0170a8a768bc",
		plugins: map[string]plugin{
			"errors": {
				options: map[string]option{
					"consolidate": {},
				},
			},
			"log": {
				options: map[string]option{
					"class": {},
				},
			},
			"health":   {},
			"autopath": {},
			"kubernetes": {
				options: map[string]option{
					"resyncperiod":       {},
					"endpoint":           {},
					"tls":                {},
					"kubeconfig":         {},
					"namespaces":         {},
					"labels":             {},
					"pods":               {},
					"endpoint_pod_names": {},
					"upstream":           {},
					"ttl":                {},
					"noendpoints":        {},
					"transfer":           {},
					"fallthrough":        {},
					"ignore":             {},
				},
			},
			"k8s_external": {
				downAction: removePlugin,
				options: map[string]option{
					"apex": {},
					"ttl":  {},
				},
			},
			"prometheus": {},
			"proxy": {
				options: map[string]option{
					"policy":       {},
					"fail_timeout": {},
					"max_fails":    {},
					"health_check": {},
					"except":       {},
					"spray":        {},
					"protocol":     {},
				},
			},
			"forward": {
				options: map[string]option{
					"except":         {},
					"force_tcp":      {},
					"prefer_udp":     {},
					"expire":         {},
					"max_fails":      {},
					"tls":            {},
					"tls_servername": {},
					"policy":         {},
					"health_check":   {},
				},
			},
			"cache": {
				options: map[string]option{
					"success":  {},
					"denial":   {},
					"prefetch": {},
				},
			},
			"loop":        {},
			"reload":      {},
			"loadbalance": {},
		},
	},
	"1.2.6": {
		nextVersion:    "1.3.0",
		priorVersion:   "1.2.5",
		k8sReleases:    []string{"1.13"},
		dockerImageSHA: "81936728011c0df9404cb70b95c17bbc8af922ec9a70d0561a5d01fefa6ffa51",
		defaultConf: `.:53 {
    errors
    health
    kubernetes * *** {
        pods insecure
        upstream
        fallthrough in-addr.arpa ip6.arpa
    }
    prometheus :9153
    proxy . *
    cache 30
    loop
    reload
    loadbalance
}`,
		plugins: map[string]plugin{
			"errors": {
				options: map[string]option{
					"consolidate": {},
				},
			},
			"log": {
				options: map[string]option{
					"class": {},
				},
			},
			"health":   {},
			"autopath": {},
			"kubernetes": {
				options: map[string]option{
					"resyncperiod":       {},
					"endpoint":           {},
					"tls":                {},
					"kubeconfig":         {},
					"namespaces":         {},
					"labels":             {},
					"pods":               {},
					"endpoint_pod_names": {},
					"upstream":           {},
					"ttl":                {},
					"noendpoints":        {},
					"transfer":           {},
					"fallthrough":        {},
					"ignore":             {},
				},
			},
			"prometheus": {},
			"proxy": {
				options: map[string]option{
					"policy":       {},
					"fail_timeout": {},
					"max_fails":    {},
					"health_check": {},
					"except":       {},
					"spray":        {},
					"protocol":     {},
				},
			},
			"forward": {
				options: map[string]option{
					"except":         {},
					"force_tcp":      {},
					"prefer_udp":     {},
					"expire":         {},
					"max_fails":      {},
					"tls":            {},
					"tls_servername": {},
					"policy":         {},
					"health_check":   {},
				},
			},
			"cache": {
				options: map[string]option{
					"success":  {},
					"denial":   {},
					"prefetch": {},
				},
			},
			"loop":        {},
			"reload":      {},
			"loadbalance": {},
		},
	},
	"1.2.5": {
		nextVersion:    "1.2.6",
		priorVersion:   "1.2.4",
		dockerImageSHA: "33c8da20b887ae12433ec5c40bfddefbbfa233d5ce11fb067122e68af30291d6",
		plugins: map[string]plugin{
			"errors": {},
			"log": {
				options: map[string]option{
					"class": {},
				},
			},
			"health":   {},
			"autopath": {},
			"kubernetes": {
				options: map[string]option{
					"resyncperiod":       {},
					"endpoint":           {},
					"tls":                {},
					"kubeconfig":         {},
					"namespaces":         {},
					"labels":             {},
					"pods":               {},
					"endpoint_pod_names": {},
					"upstream":           {},
					"ttl":                {},
					"noendpoints":        {},
					"transfer":           {},
					"fallthrough":        {},
					"ignore":             {},
				},
			},
			"prometheus": {},
			"proxy": {
				options: map[string]option{
					"policy":       {},
					"fail_timeout": {},
					"max_fails":    {},
					"health_check": {},
					"except":       {},
					"spray":        {},
					"protocol":     {},
				},
			},
			"forward": {
				options: map[string]option{
					"except":         {},
					"force_tcp":      {},
					"prefer_udp":     {},
					"expire":         {},
					"max_fails":      {},
					"tls":            {},
					"tls_servername": {},
					"policy":         {},
					"health_check":   {},
				},
			},
			"cache": {
				options: map[string]option{
					"success":  {},
					"denial":   {},
					"prefetch": {},
				},
			},
			"loop":        {},
			"reload":      {},
			"loadbalance": {},
		},
	},
	"1.2.4": {
		nextVersion:    "1.2.5",
		priorVersion:   "1.2.3",
		dockerImageSHA: "a0d40ad961a714c699ee7b61b77441d165f6252f9fb84ac625d04a8d8554c0ec",
		plugins: map[string]plugin{
			"errors": {},
			"log": {
				options: map[string]option{
					"class": {},
				},
			},
			"health":   {},
			"autopath": {},
			"kubernetes": {
				options: map[string]option{
					"resyncperiod":       {},
					"endpoint":           {},
					"tls":                {},
					"kubeconfig":         {},
					"namespaces":         {},
					"labels":             {},
					"pods":               {},
					"endpoint_pod_names": {},
					"upstream":           {},
					"ttl":                {},
					"noendpoints":        {},
					"transfer":           {},
					"fallthrough":        {},
					"ignore":             {},
				},
			},
			"prometheus": {},
			"proxy": {
				options: map[string]option{
					"policy":       {},
					"fail_timeout": {},
					"max_fails":    {},
					"health_check": {},
					"except":       {},
					"spray":        {},
					"protocol":     {},
				},
			},
			"forward": {
				options: map[string]option{
					"except":         {},
					"force_tcp":      {},
					"prefer_udp":     {},
					"expire":         {},
					"max_fails":      {},
					"tls":            {},
					"tls_servername": {},
					"policy":         {},
					"health_check":   {},
				},
			},
			"cache": {
				options: map[string]option{
					"success":  {},
					"denial":   {},
					"prefetch": {},
				},
			},
			"loop":        {},
			"reload":      {},
			"loadbalance": {},
		},
	},
	"1.2.3": {
		nextVersion:    "1.2.4",
		priorVersion:   "1.2.2",
		dockerImageSHA: "12f3cab301c826978fac736fd40aca21ac023102fd7f4aa6b4341ae9ba89e90e",
		plugins: map[string]plugin{
			"errors": {},
			"log": {
				options: map[string]option{
					"class": {},
				},
			},
			"health":   {},
			"autopath": {},
			"kubernetes": {
				options: map[string]option{
					"resyncperiod":       {},
					"endpoint":           {},
					"tls":                {},
					"kubeconfig":         {},
					"namespaces":         {},
					"labels":             {},
					"pods":               {},
					"endpoint_pod_names": {},
					"upstream":           {},
					"ttl":                {},
					"noendpoints":        {},
					"transfer":           {},
					"fallthrough":        {},
					"ignore":             {},
				},
			},
			"prometheus": {},
			"proxy": {
				options: map[string]option{
					"policy":       {},
					"fail_timeout": {},
					"max_fails":    {},
					"health_check": {},
					"except":       {},
					"spray":        {},
					"protocol":     {},
				},
			},
			"forward": {
				options: map[string]option{
					"except":         {},
					"force_tcp":      {},
					"prefer_udp":     {},
					"expire":         {},
					"max_fails":      {},
					"tls":            {},
					"tls_servername": {},
					"policy":         {},
					"health_check":   {},
				},
			},
			"cache": {
				options: map[string]option{
					"success":  {},
					"denial":   {},
					"prefetch": {},
				},
			},
			"loop":        {},
			"reload":      {},
			"loadbalance": {},
		},
	},
	"1.2.2": {
		nextVersion:    "1.2.3",
		priorVersion:   "1.2.1",
		k8sReleases:    []string{"1.12"},
		dockerImageSHA: "3e2be1cec87aca0b74b7668bbe8c02964a95a402e45ceb51b2252629d608d03a",
		defaultConf: `.:53 {
    errors
    health
    kubernetes * *** {
        pods insecure
        upstream
        fallthrough in-addr.arpa ip6.arpa
    }
    prometheus :9153
    proxy . *
    cache 30
    loop
    reload
    loadbalance
}`,
		plugins: map[string]plugin{
			"errors": {},
			"log": {
				options: map[string]option{
					"class": {},
				},
			},
			"health":   {},
			"autopath": {},
			"kubernetes": {
				options: map[string]option{
					"resyncperiod":       {},
					"endpoint":           {},
					"tls":                {},
					"namespaces":         {},
					"labels":             {},
					"pods":               {},
					"endpoint_pod_names": {},
					"upstream":           {},
					"ttl":                {},
					"noendpoints":        {},
					"transfer":           {},
					"fallthrough":        {},
					"ignore":             {},
				},
			},
			"prometheus": {},
			"proxy": {
				options: map[string]option{
					"policy":       {},
					"fail_timeout": {},
					"max_fails":    {},
					"health_check": {},
					"except":       {},
					"spray":        {},
					"protocol":     {},
				},
			},
			"forward": {
				options: map[string]option{
					"except":         {},
					"force_tcp":      {},
					"prefer_udp":     {},
					"expire":         {},
					"max_fails":      {},
					"tls":            {},
					"tls_servername": {},
					"policy":         {},
					"health_check":   {},
				},
			},
			"cache": {
				options: map[string]option{
					"success":  {},
					"denial":   {},
					"prefetch": {},
				},
			},
			"loop":        {},
			"reload":      {},
			"loadbalance": {},
		},
	},
	"1.2.1": {
		nextVersion:    "1.2.2",
		priorVersion:   "1.2.0",
		dockerImageSHA: "fb129c6a7c8912bc6d9cc4505e1f9007c5565ceb1aa6369750e60cc79771a244",
		plugins: map[string]plugin{
			"errors": {},
			"log": {
				options: map[string]option{
					"class": {},
				},
			},
			"health":   {},
			"autopath": {},
			"kubernetes": {
				options: map[string]option{
					"resyncperiod":       {},
					"endpoint":           {},
					"tls":                {},
					"namespaces":         {},
					"labels":             {},
					"pods":               {},
					"endpoint_pod_names": {},
					"upstream":           {},
					"ttl":                {},
					"noendpoints":        {},
					"transfer":           {},
					"fallthrough":        {},
					"ignore":             {},
				},
			},
			"prometheus": {},
			"proxy": {
				options: map[string]option{
					"policy":       {},
					"fail_timeout": {},
					"max_fails":    {},
					"health_check": {},
					"except":       {},
					"spray":        {},
					"protocol":     {},
				},
			},
			"forward": {
				options: map[string]option{
					"except":         {},
					"force_tcp":      {},
					"prefer_udp":     {},
					"expire":         {},
					"max_fails":      {},
					"tls":            {},
					"tls_servername": {},
					"policy":         {},
					"health_check":   {},
				},
			},
			"cache": {
				options: map[string]option{
					"success":  {},
					"denial":   {},
					"prefetch": {},
				},
			},
			"loop": {
				status: newdefault,
				add: func(s *corefile.Server) (*corefile.Server, error) {
					return addToForwardingServerBlocks(s, &corefile.Plugin{Name: "loop"})
				},
				downAction: removePlugin,
			},
			"reload":      {},
			"loadbalance": {},
		},
	},
	"1.2.0": {
		nextVersion:    "1.2.1",
		priorVersion:   "1.1.4",
		dockerImageSHA: "ae69a32f8cc29a3e2af9628b6473f24d3e977950a2cb62ce8911478a61215471",
		plugins: map[string]plugin{
			"errors": {},
			"log": {
				options: map[string]option{
					"class": {},
				},
			},
			"health":   {},
			"autopath": {},
			"kubernetes": {
				options: map[string]option{
					"resyncperiod":       {},
					"endpoint":           {},
					"tls":                {},
					"namespaces":         {},
					"labels":             {},
					"pods":               {},
					"endpoint_pod_names": {},
					"upstream":           {},
					"ttl":                {},
					"noendpoints":        {},
					"transfer":           {},
					"fallthrough":        {},
					"ignore":             {},
				},
			},
			"prometheus": {},
			"proxy": {
				options: map[string]option{
					"policy":       {},
					"fail_timeout": {},
					"max_fails":    {},
					"health_check": {},
					"except":       {},
					"spray":        {},
					"protocol": {
						status: removed,
						action: proxyRemoveHttpsGoogleProtocol,
					},
				},
			},
			"forward": {
				options: map[string]option{
					"except":         {},
					"force_tcp":      {},
					"prefer_udp":     {},
					"expire":         {},
					"max_fails":      {},
					"tls":            {},
					"tls_servername": {},
					"policy":         {},
					"health_check":   {},
				},
			},
			"cache": {
				options: map[string]option{
					"success":  {},
					"denial":   {},
					"prefetch": {},
				},
			},
			"reload":      {},
			"loadbalance": {},
		},
	},
	"1.1.4": {
		nextVersion:    "1.2.0",
		priorVersion:   "1.1.3",
		dockerImageSHA: "463c7021141dd3bfd4a75812f4b735ef6aadc0253a128f15ffe16422abe56e50",
		plugins: map[string]plugin{
			"errors": {},
			"log": {
				options: map[string]option{
					"class": {},
				},
			},
			"health":   {},
			"autopath": {},
			"kubernetes": {
				options: map[string]option{
					"resyncperiod":       {},
					"endpoint":           {},
					"tls":                {},
					"namespaces":         {},
					"labels":             {},
					"pods":               {},
					"endpoint_pod_names": {},
					"upstream":           {},
					"ttl":                {},
					"noendpoints":        {},
					"transfer":           {},
					"fallthrough":        {},
					"ignore":             {},
				},
			},
			"prometheus": {},
			"proxy": {
				options: map[string]option{
					"policy":       {},
					"fail_timeout": {},
					"max_fails":    {},
					"health_check": {},
					"except":       {},
					"spray":        {},
					"protocol": {
						status: ignored,
						action: proxyRemoveHttpsGoogleProtocol,
					},
				},
			},
			"forward": {
				options: map[string]option{
					"except":         {},
					"force_tcp":      {},
					"expire":         {},
					"max_fails":      {},
					"tls":            {},
					"tls_servername": {},
					"policy":         {},
					"health_check":   {},
				},
			},
			"cache": {
				options: map[string]option{
					"success":  {},
					"denial":   {},
					"prefetch": {},
				},
			},
			"reload":      {},
			"loadbalance": {},
		},
	},
	"1.1.3": {
		nextVersion:    "1.1.4",
		k8sReleases:    []string{"1.11"},
		dockerImageSHA: "a5dd18e048983c7401e15648b55c3ef950601a86dd22370ef5dfc3e72a108aaa",
		defaultConf: `.:53 {
    errors
    health
    kubernetes * *** {
        pods insecure
        upstream
        fallthrough in-addr.arpa ip6.arpa
    }
    prometheus :9153
    proxy . *
    cache 30
    reload
}`},
}

var proxyToForwardOptionsMigrations = map[string]option{
	"policy": {
		action: func(o *corefile.Option) (*corefile.Option, error) {
			if len(o.Args) == 2 && o.Args[1] == "least_conn" {
				o.Name = "force_tcp"
				o.Args = nil
			}
			return o, nil
		},
	},
	"except":       {},
	"fail_timeout": {action: removeOption},
	"max_fails":    {action: removeOption},
	"health_check": {action: removeOption},
	"spray":        {action: removeOption},
	"protocol": {
		action: func(o *corefile.Option) (*corefile.Option, error) {
			if len(o.Args) >= 2 && o.Args[1] == "force_tcp" {
				o.Name = "force_tcp"
				o.Args = nil
				return o, nil
			}
			return nil, nil
		},
	},
}

var proxyToForwardPluginAction = func(p *corefile.Plugin) (*corefile.Plugin, error) {
	return renamePlugin(p, "forward")
}

var useFirstArgumentOnly = func(o *corefile.Option) (*corefile.Option, error) {
	if len(o.Args) < 1 {
		return o, nil
	}
	o.Args = o.Args[:1]
	return o, nil
}

var proxyRemoveHttpsGoogleProtocol = func(o *corefile.Option) (*corefile.Option, error) {
	if len(o.Args) > 0 && o.Args[0] == "https_google" {
		return nil, nil
	}
	return o, nil
}

func breakForwardStubDomainsIntoServerBlocks(cf *corefile.Corefile) (*corefile.Corefile, error) {
	for _, sb := range cf.Servers {
		for j, fwd := range sb.Plugins {
			if fwd.Name != "forward" {
				continue
			}
			if len(fwd.Args) == 0 {
				return nil, errors.New("found invalid forward plugin declaration")
			}
			if fwd.Args[0] == "." {
				// dont move the default upstream
				continue
			}
			if len(sb.DomPorts) != 1 {
				return cf, errors.New("unhandled migration of multi-domain/port server block")
			}
			if sb.DomPorts[0] != "." && sb.DomPorts[0] != ".:53" {
				return cf, errors.New("unhandled migration of non-default domain/port server block")
			}

			newSb := &corefile.Server{}                // create a new server block
			newSb.DomPorts = []string{fwd.Args[0]}     // copy the forward zone to the server block domain
			fwd.Args[0] = "."                          // the plugin's zone changes to "." for brevity
			newSb.Plugins = append(newSb.Plugins, fwd) // add the plugin to its new server block

			// Add appropriate addtl plugins to new server block
			newSb.Plugins = append(newSb.Plugins, &corefile.Plugin{Name: "loop"})
			newSb.Plugins = append(newSb.Plugins, &corefile.Plugin{Name: "errors"})
			newSb.Plugins = append(newSb.Plugins, &corefile.Plugin{Name: "cache", Args: []string{"30"}})

			//add new server block to corefile
			cf.Servers = append(cf.Servers, newSb)

			//remove the forward plugin from the original server block
			sb.Plugins = append(sb.Plugins[:j], sb.Plugins[j+1:]...)
		}
	}
	return cf, nil
}
