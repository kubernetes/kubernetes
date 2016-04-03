FROM ubuntu
ENV name value
ENV name=value
ENV name=value name2=value2
ENV name="value value1"
ENV name=value\ value2
ENV name="value'quote space'value2"
ENV name='value"double quote"value2'
ENV name=value\ value2 name2=value2\ value3
ENV name="a\"b"
ENV name="a\'b"
ENV name='a\'b'
ENV name='a\'b''
ENV name='a\"b'
ENV name="''"
# don't put anything after the next line - it must be the last line of the
# Dockerfile and it must end with \
ENV name=value \
    name1=value1 \
    name2="value2a \
           value2b" \
    name3="value3a\n\"value3b\"" \
	name4="value4a\\nvalue4b" \
