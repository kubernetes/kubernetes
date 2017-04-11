## Synopsis

Static compilation of html from markdown including processing for grouping code snippets into arbitrary tabs.

## Code Example

\> bdocs-tab:kubectl Deployment Config to run 3 nginx instances (max rollback set to 10 revisions).

bdocs-tab:tab will be stripped during rendering and utilized to with CSS to show or hide the prefered tab. kubectl indicates the desired tab, since blockquotes have no specific syntax highlighting.

\`\`\`bdocs-tab:kubectl_yaml
apiVersion: extensions/v1beta1
kind: Deployment
metadata:
  name: deployment-example
spec:
  replicas: 3
  revisionHistoryLimit: 10
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.10
\`\`\`

bdocs-tab:tab_lang will be used to indicate which tab these code snippets belong to. The tab section of the string indicates the tab, while, the language is pushed beyond the underscore. During rendering, the language will be properly highlighted as if the bdoc token was omitted.

## Motivation

This is a project to extend markdown documents and render them in html with a table of contents and code snippet pane. Most projects of this variety lean heavily on front end parsing with JavaScript/jQuery. This project uses NodeJS, Marked, and highlight.js to output syntax highlighted code blocks.

With specific tokens on blockquotes and code blocks, the chunks can be placed according to their relevance. Ex: Multiple language code blocks that should be grouped under an arbitrary tab.

## Installation

Clone the repository, then add documents into documents directory. Modify the manifest.json to contain the document filenames in the order desired. The docs field is an array of objects with a filename key.

As a NodeJS program, a valid installation of node is required. Once node is installed, verify it can be run from command line.
```
node --version
```
Next, depedencies need to be installed via npm from the root of the project directory.
```
npm install
```

Once dependencies are installed, run
```
node brodoc.js
```

This will generate the index.html file, which can be opened in a browser or served.

The included node-static server can be run from the project root via
```
npm start
```

## License

Apache License Version 2.0

## FAQ

Q: Why is it named brodocs?
A: This project was born out of a collaboration with my brother to create a suitable docs app for his purposes. It was a fun name for the the two of us to use as actual brothers.