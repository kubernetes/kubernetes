// cfssljson splits out JSON with cert, csr, and key fields to separate
// files.
package main

import (
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"os"
)

func readFile(filespec string) ([]byte, error) {
	if filespec == "-" {
		return ioutil.ReadAll(os.Stdin)
	}
	return ioutil.ReadFile(filespec)
}

func writeFile(filespec, contents string, perms os.FileMode) {
	err := ioutil.WriteFile(filespec, []byte(contents), perms)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
}

// ResponseMessage represents the format of a CFSSL output for an error or message
type ResponseMessage struct {
	Code    int    `json:"int"`
	Message string `json:"message"`
}

// Response represents the format of a CFSSL output
type Response struct {
	Success  bool                   `json:"success"`
	Result   map[string]interface{} `json:"result"`
	Errors   []ResponseMessage      `json:"errors"`
	Messages []ResponseMessage      `json:"messages"`
}

type outputFile struct {
	Filename string
	Contents string
	IsBinary bool
	Perms    os.FileMode
}

func main() {
	bare := flag.Bool("bare", false, "the response from CFSSL is not wrapped in the API standard response")
	inFile := flag.String("f", "-", "JSON input")
	output := flag.Bool("stdout", false, "output the response instead of saving to a file")
	flag.Parse()

	var baseName string
	if flag.NArg() == 0 {
		baseName = "cert"
	} else {
		baseName = flag.Arg(0)
	}

	var input = map[string]interface{}{}
	var outs []outputFile
	var cert string
	var key string
	var csr string

	fileData, err := readFile(*inFile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to read input: %v\n", err)
		os.Exit(1)
	}

	if *bare {
		err = json.Unmarshal(fileData, &input)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Failed to parse input: %v\n", err)
			os.Exit(1)
		}
	} else {
		var response Response
		err = json.Unmarshal(fileData, &response)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Failed to parse input: %v\n", err)
			os.Exit(1)
		}

		if !response.Success {
			fmt.Fprintf(os.Stderr, "Request failed:\n")
			for _, msg := range response.Errors {
				fmt.Fprintf(os.Stderr, "\t%s\n", msg.Message)
			}
			os.Exit(1)
		}

		input = response.Result
	}

	if contents, ok := input["cert"]; ok {
		cert = contents.(string)
	} else if contents, ok = input["certificate"]; ok {
		cert = contents.(string)
	}
	if cert != "" {
		outs = append(outs, outputFile{
			Filename: baseName + ".pem",
			Contents: cert,
			Perms:    0664,
		})
	}

	if contents, ok := input["key"]; ok {
		key = contents.(string)
	} else if contents, ok = input["private_key"]; ok {
		key = contents.(string)
	}
	if key != "" {
		outs = append(outs, outputFile{
			Filename: baseName + "-key.pem",
			Contents: key,
			Perms:    0600,
		})
	}

	if contents, ok := input["encrypted_key"]; ok {
		encKey := contents.(string)
		outs = append(outs, outputFile{
			Filename: baseName + "-key.enc",
			Contents: encKey,
			IsBinary: true,
			Perms:    0600,
		})
	}

	if contents, ok := input["csr"]; ok {
		csr = contents.(string)
	} else if contents, ok = input["certificate_request"]; ok {
		csr = contents.(string)
	}
	if csr != "" {
		outs = append(outs, outputFile{
			Filename: baseName + ".csr",
			Contents: csr,
			Perms:    0644,
		})
	}

	if contents, ok := input["bundle"]; ok {
		outs = append(outs, outputFile{
			Filename: baseName + "-bundle.pem",
			Contents: contents.(string),
			Perms:    0644,
		})
	}

	if contents, ok := input["ocspResponse"]; ok {
		//ocspResponse is base64 encoded
		resp, err := base64.StdEncoding.DecodeString(contents.(string))
		if err != nil {
			fmt.Fprintf(os.Stderr, "Failed to parse ocspResponse: %v\n", err)
			os.Exit(1)
		}
		outs = append(outs, outputFile{
			Filename: baseName + "-response.der",
			Contents: string(resp),
			IsBinary: true,
			Perms:    0644,
		})
	}

	for _, e := range outs {
		if *output {
			if e.IsBinary {
				e.Contents = base64.StdEncoding.EncodeToString([]byte(e.Contents))
			}
			fmt.Fprintf(os.Stdout, "%s\n", e.Contents)
		} else {
			writeFile(e.Filename, e.Contents, e.Perms)
		}
	}
}
