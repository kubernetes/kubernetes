## Syntax highlighting for Dockerfiles
syntax "Dockerfile" "Dockerfile[^/]*$"

## Keywords
icolor red "^(ONBUILD\s+)?(ADD|ARG|CMD|COPY|ENTRYPOINT|ENV|EXPOSE|FROM|HEALTHCHECK|LABEL|MAINTAINER|RUN|SHELL|STOPSIGNAL|USER|VOLUME|WORKDIR)[[:space:]]"

## Brackets & parenthesis
color brightgreen "(\(|\)|\[|\])"

## Double ampersand
color brightmagenta "&&"

## Comments
icolor cyan "^[[:space:]]*#.*$"

## Blank space at EOL
color ,green "[[:space:]]+$"

## Strings, single-quoted
color brightwhite "'([^']|(\\'))*'" "%[qw]\{[^}]*\}" "%[qw]\([^)]*\)" "%[qw]<[^>]*>" "%[qw]\[[^]]*\]" "%[qw]\$[^$]*\$" "%[qw]\^[^^]*\^" "%[qw]![^!]*!"

## Strings, double-quoted
color brightwhite ""([^"]|(\\"))*"" "%[QW]?\{[^}]*\}" "%[QW]?\([^)]*\)" "%[QW]?<[^>]*>" "%[QW]?\[[^]]*\]" "%[QW]?\$[^$]*\$" "%[QW]?\^[^^]*\^" "%[QW]?![^!]*!"

## Single and double quotes
color brightyellow "('|\")"
