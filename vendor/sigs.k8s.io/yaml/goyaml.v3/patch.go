/*
Copyright 2023 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package yaml

// yaml_emitter_increase_indent preserves the original signature and delegates to
// yaml_emitter_increase_indent_compact without compact-sequence indentation
func yaml_emitter_increase_indent(emitter *yaml_emitter_t, flow, indentless bool) bool {
	return yaml_emitter_increase_indent_compact(emitter, flow, indentless, false)
}

// CompactSeqIndent makes it so that '- ' is considered part of the indentation.
func (e *Encoder) CompactSeqIndent() {
	e.encoder.emitter.compact_sequence_indent = true
}

// DefaultSeqIndent makes it so that '- ' is not considered part of the indentation.
func (e *Encoder) DefaultSeqIndent() {
	e.encoder.emitter.compact_sequence_indent = false
}

// yaml_emitter_process_line_comment preserves the original signature and delegates to
// yaml_emitter_process_line_comment_linebreak passing false for linebreak
func yaml_emitter_process_line_comment(emitter *yaml_emitter_t) bool {
	return yaml_emitter_process_line_comment_linebreak(emitter, false)
}
