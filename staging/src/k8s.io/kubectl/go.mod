// This is a generated file. Do not edit directly.

module k8s.io/kubectl

go 1.12

require (
	github.com/Azure/go-ansiterm v0.0.0-20170929234023-d6e3b3328b78 // indirect
	github.com/MakeNowJust/heredoc v0.0.0-20170808103936-bb23615498cd
	github.com/docker/docker v0.7.3-0.20190327010347-be7ac8be2ae0
	github.com/liggitt/tabwriter v0.0.0-20181228230101-89fcab3d43de
	github.com/mitchellh/go-wordwrap v1.0.0
	github.com/pkg/errors v0.8.0 // indirect
	github.com/russross/blackfriday v1.5.2
	github.com/sirupsen/logrus v1.2.0 // indirect
	github.com/spf13/cobra v0.0.4
	github.com/spf13/pflag v1.0.3
	golang.org/x/sys v0.0.0-20190312061237-fead79001313
	gotest.tools v2.2.0+incompatible // indirect
	k8s.io/api v0.0.0
	k8s.io/apimachinery v0.0.0
	k8s.io/client-go v0.0.0
	k8s.io/utils v0.0.0-20190221042446-c2654d5206da
)

replace (
	golang.org/x/crypto => golang.org/x/crypto v0.0.0-20181025213731-e84da0312774
	golang.org/x/net => golang.org/x/net v0.0.0-20190206173232-65e2d4e15006
	golang.org/x/sync => golang.org/x/sync v0.0.0-20181108010431-42b317875d0f
	golang.org/x/sys => golang.org/x/sys v0.0.0-20190209173611-3b5209105503
	golang.org/x/tools => golang.org/x/tools v0.0.0-20190313210603-aa82965741a9
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/client-go => ../client-go
	k8s.io/kubectl => ../kubectl
)
