// Package translatortext implements the Azure ARM Translatortext service API version 3.0.
//
// # Introduction
//
// The Microsoft Translator Text API provides a JSON-based Web API. It provides:
//
// * Translation between any supported languages to any other supported language.
// * Translation to multiple languages in one request.
// * Transliteration to convert text from one script to another script of the same language.
// * Language detection, translation, and transliteration in one request.
// * Dictionary to lookup alternative translations of a term, to find back-translations and examples showing terms used
// in context.
// * Rich language detection.
// # Base URLs
//
// The Translator Text API is available in the following clouds:
//
// | Description | Region    | Base URL                                |
// | -------     | --------  | -------                                 |
// | Azure       | Global    | api.cognitive.microsofttranslator.com   |
// | Azure       | Europe    | api-eur.cognitive.microsofttranslator.com |
//
//
// # Authentication
//
// Subscribe to the Translator Text API, part of Azure Cognitive Services, and use your subscription key from the Azure
// portal to authenticate. You can follow the steps in
// https://docs.microsoft.com/en-us/azure/cognitive-services/translator/translator-text-how-to-signup.
//
//
// The simplest way is to pass your Azure secret key to the Translator service using the http request header
// `Ocp-Apim-Subscription-Key`.
//
// If you prefer using a short-lived authentication, you may use your secret key to obtain an authorization token from
// the token service. In that case you pass the authorization token to the Translator service using the `Authorization`
// request header. To obtain an authorization token, make a `POST` request to the following URL:
//
// | Environment | Authentication service URL                                |
// | ----------  | ----------                                                |
// | Azure       | `https://api.cognitive.microsoft.com/sts/v1.0/issueToken` |
//
// Here are example requests to obtain a token with a lifetime of 10 minutes, given a secret key:
//
// ```python
// // Pass secret key using header
// curl --header 'Ocp-Apim-Subscription-Key: <your-key>' --data ""
// 'https://api.cognitive.microsoft.com/sts/v1.0/issueToken'
// // Pass secret key using query string parameter
// curl --data "" 'https://api.cognitive.microsoft.com/sts/v1.0/issueToken?Subscription-Key=<your-key>'
// ```
//
// A successful request returns the encoded access token as plain text in the response body. The valid token is passed
// to the Translator service as a bearer token in the Authorization.
//
// ```
// Authorization: Bearer <Base64-access_token>
// ```
//
// An authentication token is valid for 10 minutes. The token should be re-used when making multiple calls to the
// Translator APIs. If you make requests to the Translator API over an extended period of time,  you  must request a
// new access token at regular intervals before the token expires, for instance every 9 minutes.
//
// To summarize, a client request to the Translator API will include one authorization header taken from the following
// table:
//
// | Headers       | Description  |
// | ----------    | ----------   |
// | Ocp-Apim-Subscription-key    | Use with Cognitive Services subscription if you are passing your secret key.
// The value is the Azure secret key for your subscription to Translator Text API.                         |
// | Authorization                | Use with Cognitive Services subscription if you are passing an authentication
// token. The value is the Bearer token: `Bearer <token>`.       |
//
// ## All-in-one subscription
// The last authentication option is to use a Cognitive Service’s all-in-one subscription. This allows you to use a
// single secret key to authenticate requests for multiple services.
// When you use an all-in-one secret key, you must include two authentication headers with your request. The first
// passes the secret key, the second specifies the region associated with your subscription.
// `Ocp-Api-Subscription-Key` `Ocp-Apim-Subscription-Region`
// If you pass the secret key in the query string with the parameter `Subscription-Key`, then you must specify the
// region with query parameter `Subscription-Region`.
// If you use a bearer token, you must obtain the token from the region endpoint:
// `https://<your-region>.api.cognitive.microsoft.com/sts/v1.0/issueToken`.
//
// Available regions are: `australiaeast`, `brazilsouth`, `canadacentral`, `centralindia`, `centraluseuap`, `eastasia`,
// `eastus`, `eastus2`, `japaneast`, `northeurope`, `southcentralus`, `southeastasia`, `uksouth`, `westcentralus`,
// `westeurope`, `westus`, and `westus2`.
//
// Region is required for the all-in-one Text API subscription.
//
//
// # Errors
//
// A standard error response is a JSON object with name/value pair named `error`. The value is also a JSON object with
// properties:
// * `code`: A server-defined error code.
// * `message`: A string giving a human-readable representation of the error.
//
// For example, a customer with a free trial subscription receives the following error once the free quota is
// exhausted:
//
// ```json
// {
// "error": {
// "code":403000,
// "message":"The subscription has exceeded its free quota."
// }
// }
// ```
// # Enter your subscription keys to try out Microsoft Translator.
// Select the `Authorize` button and enter your Microsoft Translator subscription key, OR your `all in one Cognitive
// Services` subscription key. If you are using the all in one Cognitive Services key you will need to also enter your
// subscription region.
// ## Available regions are:
//
// `australiaeast`, `brazilsouth`, `canadacentral`, `centralindia`, `centraluseuap`, `eastasia`, `eastus`, `eastus2`,
// `japaneast`, `northeurope`, `southcentralus`, `southeastasia`, `uksouth`, `westcentralus`, `westeurope`, `westus`,
// `westus2`.
//
package translatortext

// Copyright (c) Microsoft and contributors.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Code generated by Microsoft (R) AutoRest Code Generator.
// Changes may cause incorrect behavior and will be lost if the code is regenerated.

import (
	"github.com/Azure/go-autorest/autorest"
)

// BaseClient is the base client for Translatortext.
type BaseClient struct {
	autorest.Client
	Endpoint string
}

// New creates an instance of the BaseClient client.
func New(endpoint string) BaseClient {
	return NewWithoutDefaults(endpoint)
}

// NewWithoutDefaults creates an instance of the BaseClient client.
func NewWithoutDefaults(endpoint string) BaseClient {
	return BaseClient{
		Client:   autorest.NewClientWithUserAgent(UserAgent()),
		Endpoint: endpoint,
	}
}
