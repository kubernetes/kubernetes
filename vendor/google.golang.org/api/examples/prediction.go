// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"google.golang.org/api/googleapi"
	prediction "google.golang.org/api/prediction/v1.6"
)

func init() {
	scopes := []string{
		prediction.DevstorageFullControlScope,
		prediction.DevstorageReadOnlyScope,
		prediction.DevstorageReadWriteScope,
		prediction.PredictionScope,
	}
	registerDemo("prediction", strings.Join(scopes, " "), predictionMain)
}

type predictionType struct {
	api              *prediction.Service
	projectNumber    string
	bucketName       string
	trainingFileName string
	modelName        string
}

// This example demonstrates calling the Prediction API.
// Training data is uploaded to a pre-created Google Cloud Storage Bucket and
// then the Prediction API is called to train a model based on that data.
// After a few minutes, the model should be completely trained and ready
// for prediction. At that point, text is sent to the model and the Prediction
// API attempts to classify the data, and the results are printed out.
//
// To get started, follow the instructions found in the "Hello Prediction!"
// Getting Started Guide located here:
// https://developers.google.com/prediction/docs/hello_world
//
// Example usage:
//   go-api-demo -clientid="my-clientid" -secret="my-secret" prediction
//       my-project-number my-bucket-name my-training-filename my-model-name
//
// Example output:
//   Predict result: language=Spanish
//   English Score: 0.000000
//   French Score: 0.000000
//   Spanish Score: 1.000000
//   analyze: output feature text=&{157 English}
//   analyze: output feature text=&{149 French}
//   analyze: output feature text=&{100 Spanish}
//   feature text count=406
func predictionMain(client *http.Client, argv []string) {
	if len(argv) != 4 {
		fmt.Fprintln(os.Stderr,
			"Usage: prediction project_number bucket training_data model_name")
		return
	}

	api, err := prediction.New(client)
	if err != nil {
		log.Fatalf("unable to create prediction API client: %v", err)
	}

	t := &predictionType{
		api:              api,
		projectNumber:    argv[0],
		bucketName:       argv[1],
		trainingFileName: argv[2],
		modelName:        argv[3],
	}

	t.trainModel()
	t.predictModel()
}

func (t *predictionType) trainModel() {
	// First, check to see if our trained model already exists.
	res, err := t.api.Trainedmodels.Get(t.projectNumber, t.modelName).Do()
	if err != nil {
		if ae, ok := err.(*googleapi.Error); ok && ae.Code != http.StatusNotFound {
			log.Fatalf("error getting trained model: %v", err)
		}
		log.Printf("Training model not found, creating new model.")
		res, err = t.api.Trainedmodels.Insert(t.projectNumber, &prediction.Insert{
			Id:                  t.modelName,
			StorageDataLocation: filepath.Join(t.bucketName, t.trainingFileName),
		}).Do()
		if err != nil {
			log.Fatalf("unable to create trained model: %v", err)
		}
	}
	if res.TrainingStatus != "DONE" {
		// Wait for the trained model to finish training.
		fmt.Printf("Training model. Please wait and re-run program after a few minutes.")
		os.Exit(0)
	}
}

func (t *predictionType) predictModel() {
	// Model has now been trained.  Predict with it.
	input := &prediction.Input{
		Input: &prediction.InputInput{
			CsvInstance: []interface{}{
				"Hola, con quien hablo",
			},
		},
	}
	res, err := t.api.Trainedmodels.Predict(t.projectNumber, t.modelName, input).Do()
	if err != nil {
		log.Fatalf("unable to get trained prediction: %v", err)
	}
	fmt.Printf("Predict result: language=%v\n", res.OutputLabel)
	for _, m := range res.OutputMulti {
		fmt.Printf("%v Score: %v\n", m.Label, m.Score)
	}

	// Now analyze the model.
	an, err := t.api.Trainedmodels.Analyze(t.projectNumber, t.modelName).Do()
	if err != nil {
		log.Fatalf("unable to analyze trained model: %v", err)
	}
	for _, f := range an.DataDescription.OutputFeature.Text {
		fmt.Printf("analyze: output feature text=%v\n", f)
	}
	for _, f := range an.DataDescription.Features {
		fmt.Printf("feature text count=%v\n", f.Text.Count)
	}
}
