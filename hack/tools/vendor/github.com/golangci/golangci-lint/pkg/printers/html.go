package printers

import (
	"context"
	"fmt"
	"html/template"
	"io"
	"strings"

	"github.com/golangci/golangci-lint/pkg/result"
)

const templateContent = `<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>golangci-lint</title>
    <link rel="shortcut icon" type="image/png" href="https://golangci-lint.run/favicon-32x32.png">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/bulma/0.9.2/css/bulma.min.css"
          integrity="sha512-byErQdWdTqREz6DLAA9pCnLbdoGGhXfU6gm1c8bkf7F51JVmUBlayGe2A31VpXWQP+eiJ3ilTAZHCR3vmMyybA=="
          crossorigin="anonymous" referrerpolicy="no-referrer"/>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/10.7.2/styles/default.min.css"
          integrity="sha512-kZqGbhf9JTB4bVJ0G8HCkqmaPcRgo88F0dneK30yku5Y/dep7CZfCnNml2Je/sY4lBoqoksXz4PtVXS4GHSUzQ=="
          crossorigin="anonymous" referrerpolicy="no-referrer"/>
    <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/10.7.2/highlight.min.js"
            integrity="sha512-s+tOYYcC3Jybgr9mVsdAxsRYlGNq4mlAurOrfNuGMQ/SCofNPu92tjE7YRZCsdEtWL1yGkqk15fU/ark206YTg=="
            crossorigin="anonymous" referrerpolicy="no-referrer"></script>
    <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/10.7.2/languages/go.min.js"
            integrity="sha512-+UYV2NyyynWEQcZ4sMTKmeppyV331gqvMOGZ61/dqc89Tn1H40lF05ACd03RSD9EWwGutNwKj256mIR8waEJBQ=="
            crossorigin="anonymous" referrerpolicy="no-referrer"></script>
    <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/react/17.0.2/umd/react.production.min.js"
            integrity="sha512-qlzIeUtTg7eBpmEaS12NZgxz52YYZVF5myj89mjJEesBd/oE9UPsYOX2QAXzvOAZYEvQohKdcY8zKE02ifXDmA=="
            crossorigin="anonymous" referrerpolicy="no-referrer"></script>
    <script type="text/javascript"
            src="https://cdnjs.cloudflare.com/ajax/libs/react-dom/17.0.2/umd/react-dom.production.min.js"
            integrity="sha512-9jGNr5Piwe8nzLLYTk8QrEMPfjGU0px80GYzKZUxi7lmCfrBjtyCc1V5kkS5vxVwwIB7Qpzc7UxLiQxfAN30dw=="
            crossorigin="anonymous" referrerpolicy="no-referrer"></script>
    <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/babel-standalone/6.26.0/babel.min.js"
            integrity="sha512-kp7YHLxuJDJcOzStgd6vtpxr4ZU9kjn77e6dBsivSz+pUuAuMlE2UTdKB7jjsWT84qbS8kdCWHPETnP/ctrFsA=="
            crossorigin="anonymous" referrerpolicy="no-referrer"></script>
</head>
<body>
<section class="section">
    <div class="container">
        <div id="content"></div>
    </div>
</section>
<script>
    const data = {{ . }};
</script>
<script type="text/babel">
  class Highlight extends React.Component {
    componentDidMount() {
      hljs.highlightElement(ReactDOM.findDOMNode(this));
    }

    render() {
      return <pre className="go"><code>{this.props.code}</code></pre>;
    }
  }

  class Issue extends React.Component {
    render() {
      return (
        <div className="issue box">
          <div>
            <div className="columns">
              <div className="column is-four-fifths">
                <h5 className="title is-5 has-text-danger-dark">{this.props.data.Title}</h5>
              </div>
              <div className="column is-one-fifth">
                <h6 className="title is-6">{this.props.data.Linter}</h6>
              </div>
            </div>
            <strong>{this.props.data.Pos}</strong>
          </div>
          <div className="highlight">
            <Highlight code={this.props.data.Code}/>
          </div>
        </div>
      );
    }
  }

  class Issues extends React.Component {
    render() {
      if (!this.props.data.Issues || this.props.data.Issues.length === 0) {
        return (
          <div>
            <div className="notification">
              No issues found!
            </div>
          </div>
        );
      }

      return (
        <div className="issues">
          {this.props.data.Issues.map(issue => (<Issue data={issue}/>))}
        </div>
      );
    }
  }

  ReactDOM.render(
    <div className="content">
      <div className="columns is-centered">
        <div className="column is-three-quarters">
          <Issues data={data}/>
        </div>
      </div>
    </div>,
    document.getElementById("content")
  );
</script>
</body>
</html>`

type htmlIssue struct {
	Title  string
	Pos    string
	Linter string
	Code   string
}

type HTML struct {
	w io.Writer
}

func NewHTML(w io.Writer) *HTML {
	return &HTML{w: w}
}

func (p HTML) Print(_ context.Context, issues []result.Issue) error {
	var htmlIssues []htmlIssue

	for i := range issues {
		pos := fmt.Sprintf("%s:%d", issues[i].FilePath(), issues[i].Line())
		if issues[i].Pos.Column != 0 {
			pos += fmt.Sprintf(":%d", issues[i].Pos.Column)
		}

		htmlIssues = append(htmlIssues, htmlIssue{
			Title:  strings.TrimSpace(issues[i].Text),
			Pos:    pos,
			Linter: issues[i].FromLinter,
			Code:   strings.Join(issues[i].SourceLines, "\n"),
		})
	}

	t, err := template.New("golangci-lint").Parse(templateContent)
	if err != nil {
		return err
	}

	return t.Execute(p.w, struct{ Issues []htmlIssue }{Issues: htmlIssues})
}
