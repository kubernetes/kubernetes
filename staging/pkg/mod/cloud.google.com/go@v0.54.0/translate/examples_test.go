// Copyright 2016 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package translate_test

import (
	"context"
	"fmt"

	"cloud.google.com/go/translate"
	"golang.org/x/text/language"
)

func Example_NewClient() {
	ctx := context.Background()
	client, err := translate.NewClient(ctx)
	if err != nil {
		// TODO: handle error.
	}
	// Use the client.

	// Close the client when finished.
	if err := client.Close(); err != nil {
		// TODO: handle error.
	}
}

func Example_Translate() {
	ctx := context.Background()
	client, err := translate.NewClient(ctx)
	if err != nil {
		// TODO: handle error.
	}
	translations, err := client.Translate(ctx,
		[]string{"Le singe est sur la branche"}, language.English,
		&translate.Options{
			Source: language.French,
			Format: translate.Text,
		})
	if err != nil {
		// TODO: handle error.
	}
	fmt.Println(translations[0].Text)
}

func Example_DetectLanguage() {
	ctx := context.Background()
	client, err := translate.NewClient(ctx)
	if err != nil {
		// TODO: handle error.
	}
	ds, err := client.DetectLanguage(ctx, []string{"Today is Monday"})
	if err != nil {
		// TODO: handle error.
	}
	fmt.Println(ds)
}

func Example_SupportedLanguages() {
	ctx := context.Background()
	client, err := translate.NewClient(ctx)
	if err != nil {
		// TODO: handle error.
	}
	langs, err := client.SupportedLanguages(ctx, language.English)
	if err != nil {
		// TODO: handle error.
	}
	fmt.Println(langs)
}
