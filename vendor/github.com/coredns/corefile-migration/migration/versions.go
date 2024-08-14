package migration

import (
	"github.com/coredns/corefile-migration/migration/corefile"
)

// release holds information pertaining to a single CoreDNS release
type release struct {
	k8sReleases    []string          // a list of K8s versions that deploy this CoreDNS release by default
	nextVersion    string            // the next CoreDNS version
	priorVersion   string            // the prior CoreDNS version
	dockerImageSHA string            // the docker image SHA for this release
	plugins        map[string]plugin // map of plugins with deprecation status and migration actions for this release

	// pre/postProcess are processing actions to take on the corefile as a whole.  Used for complex migration
	//   tasks that dont fit well into the modular plugin/option migration framework. For example, when the
	//   action on a plugin would need to extend beyond the scope of that plugin (affecting other plugins, or
	//   server blocks, etc). e.g. Splitting plugins out into separate server blocks.
	preProcess  corefileAction
	postProcess corefileAction

	// defaultConf holds the default Corefile template packaged with the corresponding k8sReleases.
	// Wildcards are used for fuzzy matching:
	//   "*"   matches exactly one token
	//   "***" matches 0 all remaining tokens on the line
	// Order of server blocks, plugins, and namedOptions does not matter.
	// Order of arguments does matter.
	defaultConf string
}

// Versions holds a map of plugin/option migrations per CoreDNS release (since 1.1.4)
var Versions = map[string]release{
	"1.11.3": {
		priorVersion:   "1.11.1",
		dockerImageSHA: "9caabbf6238b189a65d0d6e6ac138de60d6a1c419e5a341fbbb7c78382559c6e",
		plugins: 		plugins_1_11_0,
	},
	"1.11.1": {
		nextVersion:    "1.11.3",
		priorVersion:   "1.11.0",
		dockerImageSHA: "1eeb4c7316bacb1d4c8ead65571cd92dd21e27359f0d4917f1a5822a73b75db1",
		plugins: 		plugins_1_11_0,
	},
	"1.11.0": {
		nextVersion:    "1.11.1",
		priorVersion:   "1.10.1",
		dockerImageSHA: "cc3ebb05fbdba439d2d69813f162aa204b027098c8244fb3156e6e7c0f31c548",
		plugins: 		plugins_1_11_0,
	},
	"1.10.1": {
		nextVersion:    "1.11.0",
		priorVersion:   "1.10.0",
		dockerImageSHA: "a0ead06651cf580044aeb0a0feba63591858fb2e43ade8c9dea45a6a89ae7e5e",
		plugins:        plugins_1_10_1,
	},
	"1.10.0": {
		nextVersion:    "1.10.1",
		priorVersion:   "1.9.4",
		dockerImageSHA: "017727efcfeb7d053af68e51436ce8e65edbc6ca573720afb4f79c8594036955",
		plugins:        plugins_1_9_3,
	},
	"1.9.4": {
		nextVersion:    "1.10.0",
		priorVersion:   "1.9.3",
		dockerImageSHA: "b82e294de6be763f73ae71266c8f5466e7e03c69f3a1de96efd570284d35bb18",
		plugins:        plugins_1_9_3,
	},
	"1.9.3": {
		nextVersion:    "1.9.4",
		priorVersion:   "1.9.2",
		dockerImageSHA: "8e352a029d304ca7431c6507b56800636c321cb52289686a581ab70aaa8a2e2a",
		plugins:        plugins_1_9_3,
	},
	"1.9.2": {
		nextVersion:    "1.9.3",
		priorVersion:   "1.9.1",
		dockerImageSHA: "27340bfb3d563684973da8222bfed30c8b38e211d39e6dc2e632d0beef4cdca0",
		plugins:        plugins_1_8_3,
	},
	"1.9.1": {
		nextVersion:    "1.9.2",
		priorVersion:   "1.9.0",
		dockerImageSHA: "d5a7db9ab4cb3efc22a08707385c54c328db3df32841d6c4a8ae78f102f1f49a",
		plugins:        plugins_1_8_3,
	},
	"1.9.0": {
		nextVersion:    "1.9.1",
		priorVersion:   "1.8.7",
		dockerImageSHA: "0f101fabf4b63883d4529435f75b1e8816dcc8915e8fa7d28aa6e50a15e9ea6a",
		plugins:        plugins_1_8_3,
	},
	"1.8.7": {
		nextVersion:    "1.9.0",
		priorVersion:   "1.8.6",
		dockerImageSHA: "58508c172b14716350dc5185baefd78265a703514281d309d1d54aa1b721ad68",
		plugins:        plugins_1_8_3,
	},
	"1.8.6": {
		nextVersion:    "1.8.7",
		priorVersion:   "1.8.5",
		dockerImageSHA: "5b6ec0d6de9baaf3e92d0f66cd96a25b9edbce8716f5f15dcd1a616b3abd590e",
		plugins:        plugins_1_8_3,
	},
	"1.8.5": {
		nextVersion:    "1.8.6",
		priorVersion:   "1.8.4",
		dockerImageSHA: "43a9f52f5dce39bf1816afe6141724cc2d08811e466dd46e6628c925e2419bdc",
		plugins:        plugins_1_8_3,
	},
	"1.8.4": {
		nextVersion:    "1.8.5",
		priorVersion:   "1.8.3",
		dockerImageSHA: "6e5a02c21641597998b4be7cb5eb1e7b02c0d8d23cce4dd09f4682d463798890",
		plugins:        plugins_1_8_3,
	},
	"1.8.3": {
		nextVersion:    "1.8.4",
		priorVersion:   "1.8.0", // CoreDNS 1.8.2 is not a valid version and 1.8.1 docker images were never released.
		dockerImageSHA: "642ff9910da6ea9a8624b0234eef52af9ca75ecbec474c5507cb096bdfbae4e5",
		plugins:        plugins_1_8_3,
	},
	"1.8.0": {
		nextVersion:    "1.8.3", // CoreDNS 1.8.2 is not a valid version and 1.8.1 docker images were never released.
		priorVersion:   "1.7.1",
		k8sReleases:    []string{"1.21"},
		dockerImageSHA: "cc8fb77bc2a0541949d1d9320a641b82fd392b0d3d8145469ca4709ae769980e",
		plugins: map[string]plugin{
			"errors":       plugins["errors"]["v2"],
			"log":          plugins["log"]["v1"],
			"health":       plugins["health"]["v1"],
			"ready":        {},
			"autopath":     {},
			"kubernetes":   plugins["kubernetes"]["v8 remove transfer option"],
			"k8s_external": plugins["k8s_external"]["v1"],
			"prometheus":   {},
			"forward":      plugins["forward"]["v3"],
			"cache":        plugins["cache"]["v1"],
			"loop":         {},
			"reload":       {},
			"loadbalance":  {},
			"hosts":        plugins["hosts"]["v1"],
			"rewrite":      plugins["rewrite"]["v2"],
			"transfer":     plugins["transfer"]["v1"],
		},
		preProcess: copyKubernetesTransferOptToPlugin,
	},
	"1.7.1": {
		nextVersion:    "1.8.0",
		priorVersion:   "1.7.0",
		dockerImageSHA: "4a6e0769130686518325b21b0c1d0688b54e7c79244d48e1b15634e98e40c6ef",
		plugins: map[string]plugin{
			"errors":       plugins["errors"]["v2"],
			"log":          plugins["log"]["v1"],
			"health":       plugins["health"]["v1"],
			"ready":        {},
			"autopath":     {},
			"kubernetes":   plugins["kubernetes"]["v7"],
			"k8s_external": plugins["k8s_external"]["v1"],
			"prometheus":   {},
			"forward":      plugins["forward"]["v3"],
			"cache":        plugins["cache"]["v1"],
			"loop":         {},
			"reload":       {},
			"loadbalance":  {},
			"hosts":        plugins["hosts"]["v1"],
			"rewrite":      plugins["rewrite"]["v2"],
		},
	},
	"1.7.0": {
		nextVersion:    "1.7.1",
		priorVersion:   "1.6.9",
		k8sReleases:    []string{"1.19", "1.20"},
		dockerImageSHA: "73ca82b4ce829766d4f1f10947c3a338888f876fbed0540dc849c89ff256e90c",
		defaultConf: `.:53 {
    errors
    health {
        lameduck 5s
    }
    ready
    kubernetes * *** {
        pods insecure
        fallthrough in-addr.arpa ip6.arpa
        ttl 30
    }
    prometheus :9153
    forward . * {
        max_concurrent 1000
    }
    cache 30
    loop
    reload
    loadbalance
}`,
		plugins: map[string]plugin{
			"errors":       plugins["errors"]["v2"],
			"log":          plugins["log"]["v1"],
			"health":       plugins["health"]["v1"],
			"ready":        {},
			"autopath":     {},
			"kubernetes":   plugins["kubernetes"]["v7"],
			"k8s_external": plugins["k8s_external"]["v1"],
			"prometheus":   {},
			"forward":      plugins["forward"]["v3 add max_concurrent"],
			"cache":        plugins["cache"]["v1"],
			"loop":         {},
			"reload":       {},
			"loadbalance":  {},
			"hosts":        plugins["hosts"]["v1"],
			"rewrite":      plugins["rewrite"]["v2"],
		},
	},
	"1.6.9": {
		nextVersion:    "1.7.0",
		priorVersion:   "1.6.7",
		dockerImageSHA: "40ee1b708e20e3a6b8e04ccd8b6b3dd8fd25343eab27c37154946f232649ae21",
		plugins: map[string]plugin{
			"errors":       plugins["errors"]["v2"],
			"log":          plugins["log"]["v1"],
			"health":       plugins["health"]["v1"],
			"ready":        {},
			"autopath":     {},
			"kubernetes":   plugins["kubernetes"]["v6"],
			"k8s_external": plugins["k8s_external"]["v1"],
			"prometheus":   {},
			"forward":      plugins["forward"]["v3"],
			"cache":        plugins["cache"]["v2"],
			"loop":         {},
			"reload":       {},
			"loadbalance":  {},
			"hosts":        plugins["hosts"]["v1"],
			"rewrite":      plugins["rewrite"]["v2"],
		},
	},
	"1.6.7": {
		nextVersion:    "1.6.9",
		priorVersion:   "1.6.6",
		k8sReleases:    []string{"1.18"},
		dockerImageSHA: "2c8d61c46f484d881db43b34d13ca47a269336e576c81cf007ca740fa9ec0800",
		defaultConf: `.:53 {
    errors
    health {
        lameduck 5s
    }
    ready
    kubernetes * *** {
        pods insecure
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
			"errors":       plugins["errors"]["v2"],
			"log":          plugins["log"]["v1"],
			"health":       plugins["health"]["v1"],
			"ready":        {},
			"autopath":     {},
			"kubernetes":   plugins["kubernetes"]["v6"],
			"k8s_external": plugins["k8s_external"]["v1"],
			"prometheus":   {},
			"forward":      plugins["forward"]["v2"],
			"cache":        plugins["cache"]["v2"],
			"loop":         {},
			"reload":       {},
			"loadbalance":  {},
			"hosts":        plugins["hosts"]["v1"],
			"rewrite":      plugins["rewrite"]["v2"],
		},
	},
	"1.6.6": {
		nextVersion:    "1.6.7",
		priorVersion:   "1.6.5",
		dockerImageSHA: "41bee6992c2ed0f4628fcef75751048927bcd6b1cee89c79f6acb63ca5474d5a",
		plugins: map[string]plugin{
			"errors":       plugins["errors"]["v2"],
			"log":          plugins["log"]["v1"],
			"health":       plugins["health"]["v1"],
			"ready":        {},
			"autopath":     {},
			"kubernetes":   plugins["kubernetes"]["v6"],
			"k8s_external": plugins["k8s_external"]["v1"],
			"prometheus":   {},
			"forward":      plugins["forward"]["v2"],
			"cache":        plugins["cache"]["v2"],
			"loop":         {},
			"reload":       {},
			"loadbalance":  {},
			"hosts":        plugins["hosts"]["v1"],
			"rewrite":      plugins["rewrite"]["v2"],
		},
	},
	"1.6.5": {
		nextVersion:    "1.6.6",
		priorVersion:   "1.6.4",
		k8sReleases:    []string{"1.17"},
		dockerImageSHA: "7ec975f167d815311a7136c32e70735f0d00b73781365df1befd46ed35bd4fe7",
		defaultConf: `.:53 {
    errors
    health {
        lameduck 5s
    }
    ready
    kubernetes * *** {
        pods insecure
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
			"errors":       plugins["errors"]["v2"],
			"log":          plugins["log"]["v1"],
			"health":       plugins["health"]["v1 add lameduck"],
			"ready":        {},
			"autopath":     {},
			"kubernetes":   plugins["kubernetes"]["v6"],
			"k8s_external": plugins["k8s_external"]["v1"],
			"prometheus":   {},
			"forward":      plugins["forward"]["v2"],
			"cache":        plugins["cache"]["v1"],
			"loop":         {},
			"reload":       {},
			"loadbalance":  {},
			"hosts":        plugins["hosts"]["v1"],
			"rewrite":      plugins["rewrite"]["v2"],
		},
	},
	"1.6.4": {
		nextVersion:    "1.6.5",
		priorVersion:   "1.6.3",
		dockerImageSHA: "493ee88e1a92abebac67cbd4b5658b4730e0f33512461442d8d9214ea6734a9b",
		plugins:        plugins_1_6_0,
	},
	"1.6.3": {
		nextVersion:    "1.6.4",
		priorVersion:   "1.6.2",
		dockerImageSHA: "cfa7236dab4e3860881fdf755880ff8361e42f6cba2e3775ae48e2d46d22f7ba",
		plugins:        plugins_1_6_0,
	},
	"1.6.2": {
		nextVersion:    "1.6.3",
		priorVersion:   "1.6.1",
		k8sReleases:    []string{"1.16"},
		dockerImageSHA: "12eb885b8685b1b13a04ecf5c23bc809c2e57917252fd7b0be9e9c00644e8ee5",
		defaultConf: `.:53 {
    errors
    health
    ready
    kubernetes * *** {
        pods insecure
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
		plugins: plugins_1_6_0,
	},
	"1.6.1": {
		nextVersion:    "1.6.2",
		priorVersion:   "1.6.0",
		dockerImageSHA: "9ae3b6fcac4ee821362277de6bd8fd2236fa7d3e19af2ef0406d80b595620a7a",
		plugins:        plugins_1_6_0,
	},
	"1.6.0": {
		nextVersion:    "1.6.1",
		priorVersion:   "1.5.2",
		dockerImageSHA: "263d03f2b889a75a0b91e035c2a14d45d7c1559c53444c5f7abf3a76014b779d",
		plugins:        plugins_1_6_0,
	},
	"1.5.2": {
		nextVersion:    "1.6.0",
		priorVersion:   "1.5.1",
		dockerImageSHA: "586d15ec14911ee680ac9c5af20ff24b9d1412fbbf0e05862ee1f5c37baa65b2",
		plugins: map[string]plugin{
			"errors":       plugins["errors"]["v2"],
			"log":          plugins["log"]["v1"],
			"health":       plugins["health"]["v1"],
			"ready":        {},
			"autopath":     {},
			"kubernetes":   plugins["kubernetes"]["v5"],
			"k8s_external": plugins["k8s_external"]["v1"],
			"prometheus":   {},
			"forward":      plugins["forward"]["v2"],
			"cache":        plugins["cache"]["v1"],
			"loop":         {},
			"reload":       {},
			"loadbalance":  {},
			"hosts":        plugins["hosts"]["v1"],
			"rewrite":      plugins["rewrite"]["v2"],
		},
	},
	"1.5.1": {
		nextVersion:    "1.5.2",
		priorVersion:   "1.5.0",
		dockerImageSHA: "451817637035535ae1fc8639753b453fa4b781d0dea557d5da5cb3c131e62ef5",
		plugins: map[string]plugin{
			"errors":       plugins["errors"]["v2"],
			"log":          plugins["log"]["v1"],
			"health":       plugins["health"]["v1"],
			"ready":        {},
			"autopath":     {},
			"kubernetes":   plugins["kubernetes"]["v5"],
			"k8s_external": plugins["k8s_external"]["v1"],
			"prometheus":   {},
			"forward":      plugins["forward"]["v2"],
			"cache":        plugins["cache"]["v1"],
			"loop":         {},
			"reload":       {},
			"loadbalance":  {},
			"hosts":        plugins["hosts"]["v1"],
			"rewrite":      plugins["rewrite"]["v2"],
		},
	},
	"1.5.0": {
		nextVersion:    "1.5.1",
		priorVersion:   "1.4.0",
		dockerImageSHA: "e83beb5e43f8513fa735e77ffc5859640baea30a882a11cc75c4c3244a737d3c",
		plugins: map[string]plugin{
			"errors": plugins["errors"]["v2"],
			"log":    plugins["log"]["v1"],
			"health": plugins["health"]["v1"],
			"ready": {
				status: SevNewDefault,
				add: func(c *corefile.Server) (*corefile.Server, error) {
					return addToKubernetesServerBlocks(c, &corefile.Plugin{Name: "ready"})
				},
				downAction: removePlugin,
			},
			"autopath":     {},
			"kubernetes":   plugins["kubernetes"]["v5"],
			"k8s_external": plugins["k8s_external"]["v1"],
			"prometheus":   {},
			"proxy":        plugins["proxy"]["removal"],
			"forward":      plugins["forward"]["v2"],
			"cache":        plugins["cache"]["v1"],
			"loop":         {},
			"reload":       {},
			"loadbalance":  {},
			"hosts":        plugins["hosts"]["v1"],
			"rewrite":      plugins["rewrite"]["v2"],
		},
		postProcess: breakForwardStubDomainsIntoServerBlocks,
	},
	"1.4.0": {
		nextVersion:    "1.5.0",
		priorVersion:   "1.3.1",
		dockerImageSHA: "70a92e9f6fc604f9b629ca331b6135287244a86612f550941193ec7e12759417",
		plugins: map[string]plugin{
			"errors":       plugins["errors"]["v2"],
			"log":          plugins["log"]["v1"],
			"health":       plugins["health"]["v1"],
			"autopath":     {},
			"kubernetes":   plugins["kubernetes"]["v4"],
			"k8s_external": plugins["k8s_external"]["v1"],
			"prometheus":   {},
			"proxy":        plugins["proxy"]["deprecation"],
			"forward":      plugins["forward"]["v2"],
			"cache":        plugins["cache"]["v1"],
			"loop":         {},
			"reload":       {},
			"loadbalance":  {},
			"hosts":        plugins["hosts"]["v1"],
			"rewrite":      plugins["rewrite"]["v2"],
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
			"errors":       plugins["errors"]["v2"],
			"log":          plugins["log"]["v1"],
			"health":       plugins["health"]["v1"],
			"autopath":     {},
			"kubernetes":   plugins["kubernetes"]["v3"],
			"k8s_external": plugins["k8s_external"]["v1"],
			"prometheus":   {},
			"proxy":        plugins["proxy"]["v2"],
			"forward":      plugins["forward"]["v2"],
			"cache":        plugins["cache"]["v1"],
			"loop":         {},
			"reload":       {},
			"loadbalance":  {},
			"hosts":        plugins["hosts"]["v1"],
			"rewrite":      plugins["rewrite"]["v2"],
		},
	},
	"1.3.0": {
		nextVersion:    "1.3.1",
		priorVersion:   "1.2.6",
		dockerImageSHA: "e030773c7fee285435ed7fc7623532ee54c4c1c4911fb24d95cd0170a8a768bc",
		plugins: map[string]plugin{
			"errors":       plugins["errors"]["v2"],
			"log":          plugins["log"]["v1"],
			"health":       plugins["health"]["v1"],
			"autopath":     {},
			"kubernetes":   plugins["kubernetes"]["v2"],
			"k8s_external": plugins["k8s_external"]["v1"],
			"prometheus":   {},
			"proxy":        plugins["proxy"]["v2"],
			"forward":      plugins["forward"]["v2"],
			"cache":        plugins["cache"]["v1"],
			"loop":         {},
			"reload":       {},
			"loadbalance":  {},
			"hosts":        plugins["hosts"]["v1"],
			"rewrite":      plugins["rewrite"]["v2"],
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
			"errors":      plugins["errors"]["v2"],
			"log":         plugins["log"]["v1"],
			"health":      plugins["health"]["v1"],
			"autopath":    {},
			"kubernetes":  plugins["kubernetes"]["v2"],
			"prometheus":  {},
			"proxy":       plugins["proxy"]["v2"],
			"forward":     plugins["forward"]["v2"],
			"cache":       plugins["cache"]["v1"],
			"loop":        {},
			"reload":      {},
			"loadbalance": {},
			"hosts":       plugins["hosts"]["v1"],
			"rewrite":     plugins["rewrite"]["v2"],
		},
	},
	"1.2.5": {
		nextVersion:    "1.2.6",
		priorVersion:   "1.2.4",
		dockerImageSHA: "33c8da20b887ae12433ec5c40bfddefbbfa233d5ce11fb067122e68af30291d6",
		plugins: map[string]plugin{
			"errors":      plugins["errors"]["v1"],
			"log":         plugins["log"]["v1"],
			"health":      plugins["health"]["v1"],
			"autopath":    {},
			"kubernetes":  plugins["kubernetes"]["v2"],
			"prometheus":  {},
			"proxy":       plugins["proxy"]["v2"],
			"forward":     plugins["forward"]["v2"],
			"cache":       plugins["cache"]["v1"],
			"loop":        {},
			"reload":      {},
			"loadbalance": {},
			"hosts":       plugins["hosts"]["v1"],
			"rewrite":     plugins["rewrite"]["v2"],
		},
	},
	"1.2.4": {
		nextVersion:    "1.2.5",
		priorVersion:   "1.2.3",
		dockerImageSHA: "a0d40ad961a714c699ee7b61b77441d165f6252f9fb84ac625d04a8d8554c0ec",
		plugins: map[string]plugin{
			"errors":      plugins["errors"]["v1"],
			"log":         plugins["log"]["v1"],
			"health":      plugins["health"]["v1"],
			"autopath":    {},
			"kubernetes":  plugins["kubernetes"]["v2"],
			"prometheus":  {},
			"proxy":       plugins["proxy"]["v2"],
			"forward":     plugins["forward"]["v2"],
			"cache":       plugins["cache"]["v1"],
			"loop":        {},
			"reload":      {},
			"loadbalance": {},
			"hosts":       plugins["hosts"]["v1"],
			"rewrite":     plugins["rewrite"]["v2"],
		},
	},
	"1.2.3": {
		nextVersion:    "1.2.4",
		priorVersion:   "1.2.2",
		dockerImageSHA: "12f3cab301c826978fac736fd40aca21ac023102fd7f4aa6b4341ae9ba89e90e",
		plugins: map[string]plugin{
			"errors":      plugins["errors"]["v1"],
			"log":         plugins["log"]["v1"],
			"health":      plugins["health"]["v1"],
			"autopath":    {},
			"kubernetes":  plugins["kubernetes"]["v2"],
			"prometheus":  {},
			"proxy":       plugins["proxy"]["v2"],
			"forward":     plugins["forward"]["v2"],
			"cache":       plugins["cache"]["v1"],
			"loop":        {},
			"reload":      {},
			"loadbalance": {},
			"hosts":       plugins["hosts"]["v1"],
			"rewrite":     plugins["rewrite"]["v2"],
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
			"errors":      plugins["errors"]["v1"],
			"log":         plugins["log"]["v1"],
			"health":      plugins["health"]["v1"],
			"autopath":    {},
			"kubernetes":  plugins["kubernetes"]["v1"],
			"prometheus":  {},
			"proxy":       plugins["proxy"]["v2"],
			"forward":     plugins["forward"]["v2"],
			"cache":       plugins["cache"]["v1"],
			"loop":        {},
			"reload":      {},
			"loadbalance": {},
			"hosts":       plugins["hosts"]["v1"],
			"rewrite":     plugins["rewrite"]["v1"],
		},
	},
	"1.2.1": {
		nextVersion:    "1.2.2",
		priorVersion:   "1.2.0",
		dockerImageSHA: "fb129c6a7c8912bc6d9cc4505e1f9007c5565ceb1aa6369750e60cc79771a244",
		plugins: map[string]plugin{
			"errors":     plugins["errors"]["v1"],
			"log":        plugins["log"]["v1"],
			"health":     plugins["health"]["v1"],
			"autopath":   {},
			"kubernetes": plugins["kubernetes"]["v1"],
			"prometheus": {},
			"proxy":      plugins["proxy"]["v2"],
			"forward":    plugins["forward"]["v2"],
			"cache":      plugins["cache"]["v1"],
			"loop": {
				status: SevNewDefault,
				add: func(s *corefile.Server) (*corefile.Server, error) {
					return addToForwardingServerBlocks(s, &corefile.Plugin{Name: "loop"})
				},
				downAction: removePlugin,
			},
			"reload":      {},
			"loadbalance": {},
			"hosts":       plugins["hosts"]["v1"],
			"rewrite":     plugins["rewrite"]["v1"],
		},
	},
	"1.2.0": {
		nextVersion:    "1.2.1",
		priorVersion:   "1.1.4",
		dockerImageSHA: "ae69a32f8cc29a3e2af9628b6473f24d3e977950a2cb62ce8911478a61215471",
		plugins: map[string]plugin{
			"errors":      plugins["errors"]["v1"],
			"log":         plugins["log"]["v1"],
			"health":      plugins["health"]["v1"],
			"autopath":    {},
			"kubernetes":  plugins["kubernetes"]["v1"],
			"prometheus":  {},
			"proxy":       plugins["proxy"]["v2"],
			"forward":     plugins["forward"]["v2"],
			"cache":       plugins["cache"]["v1"],
			"reload":      {},
			"loadbalance": {},
			"hosts":       plugins["hosts"]["v1"],
			"rewrite":     plugins["rewrite"]["v1"],
		},
	},
	"1.1.4": {
		nextVersion:    "1.2.0",
		priorVersion:   "1.1.3",
		dockerImageSHA: "463c7021141dd3bfd4a75812f4b735ef6aadc0253a128f15ffe16422abe56e50",
		plugins: map[string]plugin{
			"errors":      plugins["errors"]["v1"],
			"log":         plugins["log"]["v1"],
			"health":      plugins["health"]["v1"],
			"autopath":    {},
			"kubernetes":  plugins["kubernetes"]["v1"],
			"prometheus":  {},
			"proxy":       plugins["proxy"]["v1"],
			"forward":     plugins["forward"]["v1"],
			"cache":       plugins["cache"]["v1"],
			"reload":      {},
			"loadbalance": {},
			"hosts":       plugins["hosts"]["v1"],
			"rewrite":     plugins["rewrite"]["v1"],
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

var plugins_1_11_0 = map[string]plugin{
	"errors":       plugins["errors"]["v3"],
	"log":          plugins["log"]["v1"],
	"health":       plugins["health"]["v1"],
	"ready":        {},
	"autopath":     {},
	"kubernetes":   plugins["kubernetes"]["v8"],
	"k8s_external": plugins["k8s_external"]["v2"], //add fallthrough option
	"prometheus":   {},
	"forward":      plugins["forward"]["v3"],
	"cache":        plugins["cache"]["v2"],
	"loop":         {},
	"reload":       {},
	"loadbalance":  {},
	"hosts":        plugins["hosts"]["v1"],
	"rewrite":      plugins["rewrite"]["v3"], // add cname_target option
	"transfer":     plugins["transfer"]["v1"],
}

var plugins_1_10_1 = map[string]plugin{
	"errors":       plugins["errors"]["v3"],
	"log":          plugins["log"]["v1"],
	"health":       plugins["health"]["v1"],
	"ready":        {},
	"autopath":     {},
	"kubernetes":   plugins["kubernetes"]["v8"],
	"k8s_external": plugins["k8s_external"]["v1"],
	"prometheus":   {},
	"forward":      plugins["forward"]["v3"],
	"cache":        plugins["cache"]["v2"], // add keepttl option
	"loop":         {},
	"reload":       {},
	"loadbalance":  {},
	"hosts":        plugins["hosts"]["v1"],
	"rewrite":      plugins["rewrite"]["v2"],
	"transfer":     plugins["transfer"]["v1"],
}

var plugins_1_9_3 = map[string]plugin{
	"errors":       plugins["errors"]["v3"], // stacktrace option added
	"log":          plugins["log"]["v1"],
	"health":       plugins["health"]["v1"],
	"ready":        {},
	"autopath":     {},
	"kubernetes":   plugins["kubernetes"]["v8"],
	"k8s_external": plugins["k8s_external"]["v1"],
	"prometheus":   {},
	"forward":      plugins["forward"]["v3"],
	"cache":        plugins["cache"]["v1"],
	"loop":         {},
	"reload":       {},
	"loadbalance":  {},
	"hosts":        plugins["hosts"]["v1"],
	"rewrite":      plugins["rewrite"]["v2"],
	"transfer":     plugins["transfer"]["v1"],
}

var plugins_1_8_3 = map[string]plugin{
	"errors":       plugins["errors"]["v2"],
	"log":          plugins["log"]["v1"],
	"health":       plugins["health"]["v1"],
	"ready":        {},
	"autopath":     {},
	"kubernetes":   plugins["kubernetes"]["v8"],
	"k8s_external": plugins["k8s_external"]["v1"],
	"prometheus":   {},
	"forward":      plugins["forward"]["v3"],
	"cache":        plugins["cache"]["v1"],
	"loop":         {},
	"reload":       {},
	"loadbalance":  {},
	"hosts":        plugins["hosts"]["v1"],
	"rewrite":      plugins["rewrite"]["v2"],
	"transfer":     plugins["transfer"]["v1"],
}

var plugins_1_6_0 = map[string]plugin{
	"errors":       plugins["errors"]["v2"],
	"log":          plugins["log"]["v1"],
	"health":       plugins["health"]["v1"],
	"ready":        {},
	"autopath":     {},
	"kubernetes":   plugins["kubernetes"]["v6"],
	"k8s_external": plugins["k8s_external"]["v1"],
	"prometheus":   {},
	"forward":      plugins["forward"]["v2"],
	"cache":        plugins["cache"]["v1"],
	"loop":         {},
	"reload":       {},
	"loadbalance":  {},
	"hosts":        plugins["hosts"]["v1"],
	"rewrite":      plugins["rewrite"]["v2"],
}
