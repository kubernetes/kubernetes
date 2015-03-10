
cdef extern from "_yaml.h":

    void malloc(int l)
    void memcpy(char *d, char *s, int l)
    int strlen(char *s)
    int PyString_CheckExact(object o)
    int PyUnicode_CheckExact(object o)
    char *PyString_AS_STRING(object o)
    int PyString_GET_SIZE(object o)
    object PyString_FromStringAndSize(char *v, int l)
    object PyUnicode_FromString(char *u)
    object PyUnicode_DecodeUTF8(char *u, int s, char *e)
    object PyUnicode_AsUTF8String(object o)
    int PY_MAJOR_VERSION

    ctypedef enum:
        SIZEOF_VOID_P
    ctypedef enum yaml_encoding_t:
        YAML_ANY_ENCODING
        YAML_UTF8_ENCODING
        YAML_UTF16LE_ENCODING
        YAML_UTF16BE_ENCODING
    ctypedef enum yaml_break_t:
        YAML_ANY_BREAK
        YAML_CR_BREAK
        YAML_LN_BREAK
        YAML_CRLN_BREAK
    ctypedef enum yaml_error_type_t:
        YAML_NO_ERROR
        YAML_MEMORY_ERROR
        YAML_READER_ERROR
        YAML_SCANNER_ERROR
        YAML_PARSER_ERROR
        YAML_WRITER_ERROR
        YAML_EMITTER_ERROR
    ctypedef enum yaml_scalar_style_t:
        YAML_ANY_SCALAR_STYLE
        YAML_PLAIN_SCALAR_STYLE
        YAML_SINGLE_QUOTED_SCALAR_STYLE
        YAML_DOUBLE_QUOTED_SCALAR_STYLE
        YAML_LITERAL_SCALAR_STYLE
        YAML_FOLDED_SCALAR_STYLE
    ctypedef enum yaml_sequence_style_t:
        YAML_ANY_SEQUENCE_STYLE
        YAML_BLOCK_SEQUENCE_STYLE
        YAML_FLOW_SEQUENCE_STYLE
    ctypedef enum yaml_mapping_style_t:
        YAML_ANY_MAPPING_STYLE
        YAML_BLOCK_MAPPING_STYLE
        YAML_FLOW_MAPPING_STYLE
    ctypedef enum yaml_token_type_t:
        YAML_NO_TOKEN
        YAML_STREAM_START_TOKEN
        YAML_STREAM_END_TOKEN
        YAML_VERSION_DIRECTIVE_TOKEN
        YAML_TAG_DIRECTIVE_TOKEN
        YAML_DOCUMENT_START_TOKEN
        YAML_DOCUMENT_END_TOKEN
        YAML_BLOCK_SEQUENCE_START_TOKEN
        YAML_BLOCK_MAPPING_START_TOKEN
        YAML_BLOCK_END_TOKEN
        YAML_FLOW_SEQUENCE_START_TOKEN
        YAML_FLOW_SEQUENCE_END_TOKEN
        YAML_FLOW_MAPPING_START_TOKEN
        YAML_FLOW_MAPPING_END_TOKEN
        YAML_BLOCK_ENTRY_TOKEN
        YAML_FLOW_ENTRY_TOKEN
        YAML_KEY_TOKEN
        YAML_VALUE_TOKEN
        YAML_ALIAS_TOKEN
        YAML_ANCHOR_TOKEN
        YAML_TAG_TOKEN
        YAML_SCALAR_TOKEN
    ctypedef enum yaml_event_type_t:
        YAML_NO_EVENT
        YAML_STREAM_START_EVENT
        YAML_STREAM_END_EVENT
        YAML_DOCUMENT_START_EVENT
        YAML_DOCUMENT_END_EVENT
        YAML_ALIAS_EVENT
        YAML_SCALAR_EVENT
        YAML_SEQUENCE_START_EVENT
        YAML_SEQUENCE_END_EVENT
        YAML_MAPPING_START_EVENT
        YAML_MAPPING_END_EVENT

    ctypedef int yaml_read_handler_t(void *data, char *buffer,
            int size, int *size_read) except 0

    ctypedef int yaml_write_handler_t(void *data, char *buffer,
            int size) except 0

    ctypedef struct yaml_mark_t:
        int index
        int line
        int column
    ctypedef struct yaml_version_directive_t:
        int major
        int minor
    ctypedef struct yaml_tag_directive_t:
        char *handle
        char *prefix

    ctypedef struct _yaml_token_stream_start_data_t:
        yaml_encoding_t encoding
    ctypedef struct _yaml_token_alias_data_t:
        char *value
    ctypedef struct _yaml_token_anchor_data_t:
        char *value
    ctypedef struct _yaml_token_tag_data_t:
        char *handle
        char *suffix
    ctypedef struct _yaml_token_scalar_data_t:
        char *value
        int length
        yaml_scalar_style_t style
    ctypedef struct _yaml_token_version_directive_data_t:
        int major
        int minor
    ctypedef struct _yaml_token_tag_directive_data_t:
        char *handle
        char *prefix
    ctypedef union _yaml_token_data_t:
        _yaml_token_stream_start_data_t stream_start
        _yaml_token_alias_data_t alias
        _yaml_token_anchor_data_t anchor
        _yaml_token_tag_data_t tag
        _yaml_token_scalar_data_t scalar
        _yaml_token_version_directive_data_t version_directive
        _yaml_token_tag_directive_data_t tag_directive
    ctypedef struct yaml_token_t:
        yaml_token_type_t type
        _yaml_token_data_t data
        yaml_mark_t start_mark
        yaml_mark_t end_mark

    ctypedef struct _yaml_event_stream_start_data_t:
        yaml_encoding_t encoding
    ctypedef struct _yaml_event_document_start_data_tag_directives_t:
        yaml_tag_directive_t *start
        yaml_tag_directive_t *end
    ctypedef struct _yaml_event_document_start_data_t:
        yaml_version_directive_t *version_directive
        _yaml_event_document_start_data_tag_directives_t tag_directives
        int implicit
    ctypedef struct _yaml_event_document_end_data_t:
        int implicit
    ctypedef struct _yaml_event_alias_data_t:
        char *anchor
    ctypedef struct _yaml_event_scalar_data_t:
        char *anchor
        char *tag
        char *value
        int length
        int plain_implicit
        int quoted_implicit
        yaml_scalar_style_t style
    ctypedef struct _yaml_event_sequence_start_data_t:
        char *anchor
        char *tag
        int implicit
        yaml_sequence_style_t style
    ctypedef struct _yaml_event_mapping_start_data_t:
        char *anchor
        char *tag
        int implicit
        yaml_mapping_style_t style
    ctypedef union _yaml_event_data_t:
        _yaml_event_stream_start_data_t stream_start
        _yaml_event_document_start_data_t document_start
        _yaml_event_document_end_data_t document_end
        _yaml_event_alias_data_t alias
        _yaml_event_scalar_data_t scalar
        _yaml_event_sequence_start_data_t sequence_start
        _yaml_event_mapping_start_data_t mapping_start
    ctypedef struct yaml_event_t:
        yaml_event_type_t type
        _yaml_event_data_t data
        yaml_mark_t start_mark
        yaml_mark_t end_mark

    ctypedef struct yaml_parser_t:
        yaml_error_type_t error
        char *problem
        int problem_offset
        int problem_value
        yaml_mark_t problem_mark
        char *context
        yaml_mark_t context_mark

    ctypedef struct yaml_emitter_t:
        yaml_error_type_t error
        char *problem

    char *yaml_get_version_string()
    void yaml_get_version(int *major, int *minor, int *patch)

    void yaml_token_delete(yaml_token_t *token)

    int yaml_stream_start_event_initialize(yaml_event_t *event,
            yaml_encoding_t encoding)
    int yaml_stream_end_event_initialize(yaml_event_t *event)
    int yaml_document_start_event_initialize(yaml_event_t *event,
            yaml_version_directive_t *version_directive,
            yaml_tag_directive_t *tag_directives_start,
            yaml_tag_directive_t *tag_directives_end,
            int implicit)
    int yaml_document_end_event_initialize(yaml_event_t *event,
            int implicit)
    int yaml_alias_event_initialize(yaml_event_t *event, char *anchor)
    int yaml_scalar_event_initialize(yaml_event_t *event,
            char *anchor, char *tag, char *value, int length,
            int plain_implicit, int quoted_implicit,
            yaml_scalar_style_t style)
    int yaml_sequence_start_event_initialize(yaml_event_t *event,
            char *anchor, char *tag, int implicit, yaml_sequence_style_t style)
    int yaml_sequence_end_event_initialize(yaml_event_t *event)
    int yaml_mapping_start_event_initialize(yaml_event_t *event,
            char *anchor, char *tag, int implicit, yaml_mapping_style_t style)
    int yaml_mapping_end_event_initialize(yaml_event_t *event)
    void yaml_event_delete(yaml_event_t *event)

    int yaml_parser_initialize(yaml_parser_t *parser)
    void yaml_parser_delete(yaml_parser_t *parser)
    void yaml_parser_set_input_string(yaml_parser_t *parser,
            char *input, int size)
    void yaml_parser_set_input(yaml_parser_t *parser,
            yaml_read_handler_t *handler, void *data)
    void yaml_parser_set_encoding(yaml_parser_t *parser,
            yaml_encoding_t encoding)
    int yaml_parser_scan(yaml_parser_t *parser, yaml_token_t *token) except *
    int yaml_parser_parse(yaml_parser_t *parser, yaml_event_t *event) except *

    int yaml_emitter_initialize(yaml_emitter_t *emitter)
    void yaml_emitter_delete(yaml_emitter_t *emitter)
    void yaml_emitter_set_output_string(yaml_emitter_t *emitter,
            char *output, int size, int *size_written)
    void yaml_emitter_set_output(yaml_emitter_t *emitter,
            yaml_write_handler_t *handler, void *data)
    void yaml_emitter_set_encoding(yaml_emitter_t *emitter,
            yaml_encoding_t encoding)
    void yaml_emitter_set_canonical(yaml_emitter_t *emitter, int canonical)
    void yaml_emitter_set_indent(yaml_emitter_t *emitter, int indent)
    void yaml_emitter_set_width(yaml_emitter_t *emitter, int width)
    void yaml_emitter_set_unicode(yaml_emitter_t *emitter, int unicode)
    void yaml_emitter_set_break(yaml_emitter_t *emitter,
            yaml_break_t line_break)
    int yaml_emitter_emit(yaml_emitter_t *emitter, yaml_event_t *event) except *
    int yaml_emitter_flush(yaml_emitter_t *emitter)

