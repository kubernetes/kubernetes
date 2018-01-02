//
// Copyright (c) 2017 The heketi Authors
//
// This file is licensed to you under your choice of the GNU Lesser
// General Public License, version 3 or any later version (LGPLv3 or
// later), or the GNU General Public License, version 2 (GPLv2), in all
// cases as published by the Free Software Foundation.
//

package kubernetes

import (
	"bytes"
	"compress/gzip"
	"fmt"
	"os"

	"github.com/boltdb/bolt"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/api/v1"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
)

var (
	inClusterConfig = restclient.InClusterConfig
	newForConfig    = func(c *restclient.Config) (clientset.Interface, error) {
		return clientset.NewForConfig(c)
	}
	getNamespace = GetNamespace
	dbSecretName = "heketi-db-backup"
)

func KubeBackupDbToSecret(db *bolt.DB) error {

	// Check if we should use another name for the heketi backup secret
	env := os.Getenv("HEKETI_KUBE_DB_SECRET_NAME")
	if len(env) != 0 {
		dbSecretName = env
	}

	// Get Kubernetes configuration
	kubeConfig, err := inClusterConfig()
	if err != nil {
		return fmt.Errorf("Unable to get kubernetes configuration: %v", err)
	}

	// Get clientset
	c, err := newForConfig(kubeConfig)
	if err != nil {
		return fmt.Errorf("Unable to get kubernetes clientset: %v", err)
	}

	// Get namespace
	ns, err := getNamespace()
	if err != nil {
		return fmt.Errorf("Unable to get namespace: %v", err)
	}

	// Create client for secrets
	secrets := c.CoreV1().Secrets(ns)
	if err != nil {
		return fmt.Errorf("Unable to get a client to kubernetes secrets: %v", err)
	}

	// Get a backup
	err = db.View(func(tx *bolt.Tx) error {
		var backup bytes.Buffer

		gz := gzip.NewWriter(&backup)
		_, err := tx.WriteTo(gz)
		if err != nil {
			return fmt.Errorf("Unable to access database: %v", err)
		}
		if err := gz.Close(); err != nil {
			return fmt.Errorf("Unable to close gzipped database: %v", err)
		}

		// Create a secret with backup
		secret := &v1.Secret{}
		secret.Kind = "Secret"
		secret.Namespace = ns
		secret.APIVersion = "v1"
		secret.ObjectMeta.Name = dbSecretName
		secret.Data = map[string][]byte{
			"heketi.db.gz": backup.Bytes(),
		}

		// Submit secret
		_, err = secrets.Create(secret)
		if apierrors.IsAlreadyExists(err) {
			// It already exists, so just update it instead
			_, err = secrets.Update(secret)
			if err != nil {
				return fmt.Errorf("Unable to update database to secret: %v", err)
			}
		} else if err != nil {
			return fmt.Errorf("Unable to create database secret: %v", err)
		}

		return nil

	})
	if err != nil {
		return fmt.Errorf("Unable to backup database to kubernetes secret: %v", err)
	}

	return nil
}
