package printers

import (
	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/printers"
)

// PrinterProvider is capable of providing a printer
// based on a given set of options
type PrinterProvider interface {
	AddFlags(cmd *cobra.Command)
	ToPrinter(outputFormat string) (printer printers.ResourcePrinter, matchedPrinter bool, err error)
}

// PrintFlags composes all known printer flag structs
// and provides a method of retrieving a known printer
// based on flag values provided.
type PrintFlags struct {
	// embedded structs (so that calling commands can bind values)
	CustomColumnsPrintFlags *CustomColumnsPrintFlags
	HumanPrintFlags         *HumanPrintFlags
	JSONYamlPrintFlags      *JSONYamlPrintFlags
	KubeTemplatePrintFlags  *KubeTemplatePrintFlags
	NamePrintFlags          *NamePrintFlags

	Providers []PrinterProvider

	OutputFormat  *string
	TemplateValue *string

	WithNamespace      bool
	AbsoluteTimestamps bool
	NoHeaders          bool
}

func (f *PrintFlags) ToPrinter() (printers.ResourcePrinter, bool, error) {
	outputFormat := ""
	if f.OutputFormat != nil {
		outputFormat = *f.OutputFormat
	}

	templateValue := ""
	if f.TemplateValue != nil {
		templateValue = *f.TemplateValue
	}

	// complete any remaining fields for embedded providers
	f.CustomColumnsPrintFlags.Complete(f.NoHeaders, templateValue)
	f.HumanPrintFlags.Complete(f.NoHeaders, f.WithNamespace, f.AbsoluteTimestamps)
	f.KubeTemplatePrintFlags.Complete(templateValue)

	for _, flags := range f.Providers {
		p, matched, err := flags.ToPrinter(outputFormat)
		if !matched {
			continue
		}
		if err != nil {
			return nil, matched, err
		}

		return p, matched, nil
	}

	return nil, false, nil
}

func (f *PrintFlags) AddFlags(cmd *cobra.Command) {
	for _, flags := range f.Providers {
		flags.AddFlags(cmd)
	}

	if f.OutputFormat != nil {
		cmd.Flags().StringVarP(f.OutputFormat, "output", "o", *f.OutputFormat, "Output format. One of: json|yaml|wide|name|custom-columns=...|custom-columns-file=...|go-template=...|go-template-file=...|jsonpath=...|jsonpath-file=... See custom columns [http://kubernetes.io/docs/user-guide/kubectl-overview/#custom-columns], golang template [http://golang.org/pkg/text/template/#pkg-overview] and jsonpath template [http://kubernetes.io/docs/user-guide/jsonpath].")
	}
	if f.TemplateValue != nil {
		cmd.Flags().StringVar(f.TemplateValue, "template", *f.TemplateValue, "Template string or path to template file to use when -o=go-template, -o=go-template-file. The template format is golang templates [http://golang.org/pkg/text/template/#pkg-overview].")
		cmd.MarkFlagFilename("template")
	}
}

func NewPrintFlags() *PrintFlags {
	noHeaders := false
	outputFormat := ""
	templateValue := ""

	flags := &PrintFlags{
		OutputFormat:  &outputFormat,
		TemplateValue: &templateValue,

		NoHeaders: noHeaders,

		CustomColumnsPrintFlags: NewCustomColumnsPrintFlags(noHeaders, templateValue),
		HumanPrintFlags:         NewHumanPrintFlags(schema.GroupKind{}, noHeaders, false, false),
		JSONYamlPrintFlags:      NewJSONYamlPrintFlags(),
		KubeTemplatePrintFlags:  NewKubeTemplatePrintFlags(templateValue),
		NamePrintFlags:          NewNamePrintFlags("", false),
	}

	flags.Providers = []PrinterProvider{
		flags.CustomColumnsPrintFlags,
		flags.HumanPrintFlags,
		flags.JSONYamlPrintFlags,
		flags.KubeTemplatePrintFlags,
		flags.NamePrintFlags,
	}

	return flags
}
