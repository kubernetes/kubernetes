//
// Copyright (c) 2015 The heketi Authors
//
// This file is licensed to you under your choice of the GNU Lesser
// General Public License, version 3 or any later version (LGPLv3 or
// later), or the GNU General Public License, version 2 (GPLv2), in all
// cases as published by the Free Software Foundation.
//

package main

import (
	"encoding/json"
	"fmt"
	"github.com/gorilla/mux"
	"github.com/heketi/heketi/apps"
	"github.com/heketi/heketi/apps/glusterfs"
	"github.com/heketi/heketi/middleware"
	"github.com/spf13/cobra"
	"github.com/urfave/negroni"
	"net/http"
	"os"
	"os/signal"
	"syscall"

	restclient "k8s.io/client-go/rest"
)

type Config struct {
	Port                 string                   `json:"port"`
	AuthEnabled          bool                     `json:"use_auth"`
	JwtConfig            middleware.JwtAuthConfig `json:"jwt"`
	BackupDbToKubeSecret bool                     `json:"backup_db_to_kube_secret"`
}

var (
	HEKETI_VERSION = "(dev)"
	configfile     string
	showVersion    bool
)

var RootCmd = &cobra.Command{
	Use:     "heketi",
	Short:   "Heketi is a restful volume management server",
	Long:    "Heketi is a restful volume management server",
	Example: "heketi --config=/config/file/path/",
	Run: func(cmd *cobra.Command, args []string) {
		fmt.Printf("Heketi %v\n", HEKETI_VERSION)
		if !showVersion {
			// Check configuration file was given
			if configfile == "" {
				fmt.Fprintln(os.Stderr, "Please provide configuration file")
				os.Exit(1)
			}
		} else {
			// Quit here if all we needed to do was show version
			os.Exit(0)
		}
	},
}

func init() {
	RootCmd.Flags().StringVar(&configfile, "config", "", "Configuration file")
	RootCmd.Flags().BoolVarP(&showVersion, "version", "v", false, "Show version")
	RootCmd.SilenceUsage = true
}

func setWithEnvVariables(options *Config) {
	// Check for user key
	env := os.Getenv("HEKETI_USER_KEY")
	if "" != env {
		options.AuthEnabled = true
		options.JwtConfig.User.PrivateKey = env
	}

	// Check for user key
	env = os.Getenv("HEKETI_ADMIN_KEY")
	if "" != env {
		options.AuthEnabled = true
		options.JwtConfig.Admin.PrivateKey = env
	}

	// Check for user key
	env = os.Getenv("HEKETI_HTTP_PORT")
	if "" != env {
		options.Port = env
	}

	env = os.Getenv("HEKETI_BACKUP_DB_TO_KUBE_SECRET")
	if "" != env {
		options.BackupDbToKubeSecret = true
	}
}

func main() {
	if err := RootCmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(-1)
	}

	// Quit here if all we needed to do was show usage/help
	if configfile == "" {
		return
	}

	// Read configuration
	fp, err := os.Open(configfile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: Unable to open config file %v: %v\n",
			configfile,
			err.Error())
		os.Exit(1)
	}
	defer fp.Close()

	configParser := json.NewDecoder(fp)
	var options Config
	if err = configParser.Decode(&options); err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: Unable to parse %v: %v\n",
			configfile,
			err.Error())
		os.Exit(1)
	}

	// Substitue values using any set environment variables
	setWithEnvVariables(&options)

	// Use negroni to add middleware.  Here we add two
	// middlewares: Recovery and Logger, which come with
	// Negroni
	n := negroni.New(negroni.NewRecovery(), negroni.NewLogger())

	// Go to the beginning of the file when we pass it
	// to the application
	fp.Seek(0, os.SEEK_SET)

	// Setup a new GlusterFS application
	var app apps.Application
	glusterfsApp := glusterfs.NewApp(fp)
	if glusterfsApp == nil {
		fmt.Fprintln(os.Stderr, "ERROR: Unable to start application")
		os.Exit(1)
	}
	app = glusterfsApp

	// Add /hello router
	router := mux.NewRouter()
	router.Methods("GET").Path("/hello").Name("Hello").HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/plain; charset=UTF-8")
			w.WriteHeader(http.StatusOK)
			fmt.Fprint(w, "Hello from Heketi")
		})

	// Create a router and do not allow any routes
	// unless defined.
	heketiRouter := mux.NewRouter().StrictSlash(true)
	err = app.SetRoutes(heketiRouter)
	if err != nil {
		fmt.Fprintln(os.Stderr, "ERROR: Unable to create http server endpoints")
		os.Exit(1)
	}

	// Load authorization JWT middleware
	if options.AuthEnabled {
		jwtauth := middleware.NewJwtAuth(&options.JwtConfig)
		if jwtauth == nil {
			fmt.Fprintln(os.Stderr, "ERROR: Missing JWT information in config file")
			os.Exit(1)
		}

		// Add Token parser
		n.Use(jwtauth)

		// Add application middleware check
		n.UseFunc(app.Auth)

		fmt.Println("Authorization loaded")
	}

	if options.BackupDbToKubeSecret {
		// Check if running in a Kubernetes environment
		_, err = restclient.InClusterConfig()
		if err == nil {
			// Load middleware to backup database
			n.UseFunc(glusterfsApp.BackupToKubernetesSecret)
		}
	}

	// Add all endpoints after the middleware was added
	n.UseHandler(heketiRouter)

	// Setup complete routing
	router.NewRoute().Handler(n)

	// Shutdown on CTRL-C signal
	// For a better cleanup, we should shutdown the server and
	signalch := make(chan os.Signal, 1)
	signal.Notify(signalch, os.Interrupt, os.Kill, syscall.SIGINT, syscall.SIGTERM)

	// Create a channel to know if the server was unable to start
	done := make(chan bool)
	go func() {
		// Start the server.
		fmt.Printf("Listening on port %v\n", options.Port)
		err = http.ListenAndServe(":"+options.Port, router)
		if err != nil {
			fmt.Printf("ERROR: HTTP Server error: %v\n", err)
		}
		done <- true
	}()

	// Block here for signals and errors from the HTTP server
	select {
	case <-signalch:
	case <-done:
	}
	fmt.Printf("Shutting down...\n")

	// Shutdown the application
	// :TODO: Need to shutdown the server
	app.Close()

}
