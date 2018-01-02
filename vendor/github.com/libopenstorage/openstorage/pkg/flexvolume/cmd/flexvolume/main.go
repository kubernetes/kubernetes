package main

import (
	"github.com/libopenstorage/openstorage/pkg/flexvolume"
	"github.com/spf13/cobra"
	"go.pedge.io/env"
	"go.pedge.io/lion/env"
	"go.pedge.io/pkg/cobra"
	"google.golang.org/grpc"
)

type appEnv struct {
	OpenstorageAddress string `env:"OPENSTORAGE_ADDRESS,default=0.0.0.0:9005"`
}

func main() {
	env.Main(do, &appEnv{})
}

func do(appEnvObj interface{}) error {
	appEnv := appEnvObj.(*appEnv)
	if err := envlion.Setup(); err != nil {
		return err
	}

	initCmd := &cobra.Command{
		Use: "init",
		Run: pkgcobra.RunFixedArgs(0, func(args []string) error {
			client, err := getClient(appEnv)
			if err != nil {
				return err
			}
			return client.Init()
		}),
	}

	attachCmd := &cobra.Command{
		Use: "attach jsonOptions",
		Run: pkgcobra.RunFixedArgs(1, func(args []string) error {
			jsonOptions, err := flexvolume.BytesToJSONOptions([]byte(args[0]))
			if err != nil {
				return err
			}
			client, err := getClient(appEnv)
			if err != nil {
				return err
			}
			return client.Attach(jsonOptions)
		}),
	}

	detachCmd := &cobra.Command{
		Use: "detach mountDevice",
		Run: pkgcobra.RunFixedArgs(1, func(args []string) error {
			client, err := getClient(appEnv)
			if err != nil {
				return err
			}
			return client.Detach(args[0], false)
		}),
	}

	mountCmd := &cobra.Command{
		Use: "mount targetMountDir mountDevice jsonOptions",
		Run: pkgcobra.RunFixedArgs(3, func(args []string) error {
			jsonOptions, err := flexvolume.BytesToJSONOptions([]byte(args[2]))
			if err != nil {
				return err
			}
			client, err := getClient(appEnv)
			if err != nil {
				return err
			}
			return client.Mount(args[0], args[1], jsonOptions)
		}),
	}

	unmountCmd := &cobra.Command{
		Use: "unmount mountDir",
		Run: pkgcobra.RunFixedArgs(1, func(args []string) error {
			client, err := getClient(appEnv)
			if err != nil {
				return err
			}
			return client.Unmount(args[0])
		}),
	}

	rootCmd := &cobra.Command{
		Use: "app",
	}
	rootCmd.AddCommand(initCmd)
	rootCmd.AddCommand(attachCmd)
	rootCmd.AddCommand(detachCmd)
	rootCmd.AddCommand(mountCmd)
	rootCmd.AddCommand(unmountCmd)
	return rootCmd.Execute()
}

func getClient(appEnv *appEnv) (flexvolume.Client, error) {
	clientConn, err := grpc.Dial(appEnv.OpenstorageAddress, grpc.WithInsecure())
	if err != nil {
		return nil, err
	}
	return flexvolume.NewClient(flexvolume.NewAPIClient(clientConn)), nil
}
