package main

const (
	defaultTemplateFile = "TEMPLATE"
	releaseNotes        = `Welcome to the release of {{.ProjectName}} {{.Version}}!
{{if .PreRelease}}
*This is a pre-release of {{.ProjectName}}*
{{- end}}

{{.Preface}}

Please try out the release binaries and report any issues at
https://github.com/{{.GithubRepo}}/issues.

{{range  $note := .Notes}}
### {{$note.Title}}

{{$note.Description}}
{{- end}}

### Contributors
{{range $contributor := .Contributors}}
* {{$contributor}}
{{- end}}

### Changes
{{range $change := .Changes}}
* {{$change.Commit}} {{$change.Description}}
{{- end}}

### Dependency Changes

Previous release can be found at [{{.Previous}}](https://github.com/{{.GithubRepo}}/releases/tag/{{.Previous}})
{{range $dep := .Dependencies}}
* {{$dep.Previous}} -> {{$dep.Commit}} **{{$dep.Name}}**
{{- end}}
`
)
