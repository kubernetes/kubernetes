#!/usr/bin/env bash

# This file contains helpful aliases for manipulating the output text to the terminal as
# well as functions for one-command augmented printing.

# os::text::reset resets the terminal output to default if it is called in a TTY
function os::text::reset() {
	if os::text::internal::is_tty; then
		tput sgr0
	fi
}
readonly -f os::text::reset

# os::text::bold sets the terminal output to bold text if it is called in a TTY
function os::text::bold() {
	if os::text::internal::is_tty; then
		tput bold
	fi
}
readonly -f os::text::bold

# os::text::red sets the terminal output to red text if it is called in a TTY
function os::text::red() {
	if os::text::internal::is_tty; then
		tput setaf 1
	fi
}
readonly -f os::text::red

# os::text::green sets the terminal output to green text if it is called in a TTY
function os::text::green() {
	if os::text::internal::is_tty; then
		tput setaf 2
	fi
}
readonly -f os::text::green

# os::text::blue sets the terminal output to blue text if it is called in a TTY
function os::text::blue() {
	if os::text::internal::is_tty; then
		tput setaf 4
	fi
}
readonly -f os::text::blue

# os::text::yellow sets the terminal output to yellow text if it is called in a TTY
function os::text::yellow() {
	if os::text::internal::is_tty; then
		tput setaf 11
	fi
}
readonly -f os::text::yellow

# os::text::clear_last_line clears the text from the last line of output to the
# terminal and leaves the cursor on that line to allow for overwriting that text
# if it is called in a TTY
function os::text::clear_last_line() {
	if os::text::internal::is_tty; then
		tput cuu 1
		tput el
	fi
}
readonly -f os::text::clear_last_line

# os::text::clear_string attempts to clear the entirety of a string from the terminal.
# If the string contains literal tabs or other characters that take up more than one
# character space in output, or if the window size is changed before this function
# is called, it will not function correctly.
# No action is taken if this is called outside of a TTY
function os::text::clear_string() {
    local -r string="$1"
    if os::text::internal::is_tty; then
        echo "${string}" | while read line; do
            # num_lines is the number of terminal lines this one line of output
            # would have taken up with the current terminal width in columns
            local num_lines=$(( ${#line} / $( tput cols ) ))
            for (( i = 0; i <= num_lines; i++ )); do
                os::text::clear_last_line
            done
        done
    fi
}

# os::text::internal::is_tty determines if we are outputting to a TTY
function os::text::internal::is_tty() {
	[[ -t 1 && -n "${TERM:-}" ]]
}
readonly -f os::text::internal::is_tty

# os::text::print_bold prints all input in bold text
function os::text::print_bold() {
	os::text::bold
	echo "${*}"
	os::text::reset
}
readonly -f os::text::print_bold

# os::text::print_red prints all input in red text
function os::text::print_red() {
	os::text::red
	echo "${*}"
	os::text::reset
}
readonly -f os::text::print_red

# os::text::print_red_bold prints all input in bold red text
function os::text::print_red_bold() {
	os::text::red
	os::text::bold
	echo "${*}"
	os::text::reset
}
readonly -f os::text::print_red_bold

# os::text::print_green prints all input in green text
function os::text::print_green() {
	os::text::green
	echo "${*}"
	os::text::reset
}
readonly -f os::text::print_green

# os::text::print_green_bold prints all input in bold green text
function os::text::print_green_bold() {
	os::text::green
	os::text::bold
	echo "${*}"
	os::text::reset
}
readonly -f os::text::print_green_bold

# os::text::print_blue prints all input in blue text
function os::text::print_blue() {
	os::text::blue
	echo "${*}"
	os::text::reset
}
readonly -f os::text::print_blue

# os::text::print_blue_bold prints all input in bold blue text
function os::text::print_blue_bold() {
	os::text::blue
	os::text::bold
	echo "${*}"
	os::text::reset
}
readonly -f os::text::print_blue_bold

# os::text::print_yellow prints all input in yellow text
function os::text::print_yellow() {
	os::text::yellow
	echo "${*}"
	os::text::reset
}
readonly -f os::text::print_yellow

# os::text::print_yellow_bold prints all input in bold yellow text
function os::text::print_yellow_bold() {
	os::text::yellow
	os::text::bold
	echo "${*}"
	os::text::reset
}
readonly -f os::text::print_yellow_bold
