<!--[metadata]>
+++
title = "Style guide for Docker documentation"
description = "Style guide for Docker documentation describing standards and conventions for contributors"
keywords = ["style, guide, docker,  documentation"]
[menu.main]
parent = "mn_opensource"
weight=100
+++
<![end-metadata]-->

# Docker documentation: style & grammar conventions

## Style standards

Over time, different publishing communities have written standards for the style
and grammar they prefer in their publications. These standards are called
[style guides](http://en.wikipedia.org/wiki/Style_guide). Generally, Docker’s
documentation uses the standards described in the
[Associated Press's (AP) style guide](http://en.wikipedia.org/wiki/AP_Stylebook). 
If a question about syntactical, grammatical, or lexical practice comes up,
refer to the AP guide first. If you don’t have a copy of (or online subscription
to) the AP guide, you can almost always find an answer to a specific question by
searching the web. If you can’t find an answer, please ask a
[maintainer](https://github.com/docker/docker/blob/master/docs/MAINTAINERS) and
we will find the answer.

That said, please don't get too hung up on using correct style. We'd rather have
you submit good information that doesn't conform to the guide than no
information at all. Docker's tech writers are always happy to help you with the
prose, and we promise not to judge or use a red pen!

> **Note:**
> The documentation is written with paragraphs wrapped at 80 column lines to
> make it easier for terminal use. You can probably set up your favorite text
> editor to do this automatically for you.

### Prose style

In general, try to write simple, declarative prose. We prefer short,
single-clause sentences and brief three-to-five sentence paragraphs. Try to
choose vocabulary that is straightforward and precise. Avoid creating new terms,
using obscure terms or, in particular, using a lot of jargon. For example, use
"use" instead of leveraging "leverage".

That said, don’t feel like you have to write for localization or for
English-as-a-second-language (ESL) speakers specifically. Assume you are writing
for an ordinary speaker of English with a basic university education. If your
prose is simple, clear, and straightforward it will translate readily.

One way to think about this is to assume Docker’s users are generally university
educated and read at at least a "16th" grade level (meaning they have a
university degree). You can use a [readability
tester](https://readability-score.com/) to help guide your judgement. For
example, the readability score for the phrase "Containers should be ephemeral"
is around the 13th grade level (first year at university), and so is acceptable.

In all cases, we prefer clear, concise communication over stilted, formal
language. Don't feel like you have to write documentation that "sounds like
technical writing."

### Metaphor and figurative language

One exception to the "don’t write directly for ESL" rule is to avoid the use of
metaphor or other
[figurative language](http://en.wikipedia.org/wiki/Literal_and_figurative_language) to
describe things. There are too many cultural and social issues that can prevent
a reader from correctly interpreting a metaphor.

## Specific conventions

Below are some specific recommendations (and a few deviations) from AP style
that we use in our docs.

### Contractions

As long as your prose does not become too slangy or informal, it's perfectly
acceptable to use contractions in our documentation. Make sure to use
apostrophes correctly.

### Use of dashes in a sentence.

Dashes refers to the en dash (–) and the em dash (—). Dashes can be used to
separate parenthetical material.

Usage Example: This is an example of a Docker client – which uses the Big Widget
to run – and does x, y, and z.

Use dashes cautiously and consider whether commas or parentheses would work just
as well. We always emphasize short, succinct sentences.

More info from the always handy [Grammar Girl site](http://www.quickanddirtytips.com/education/grammar/dashes-parentheses-and-commas).

### Pronouns

It's okay to use first and second person pronouns, especially if it lets you avoid a passive construction. Specifically, always use "we" to
refer to Docker and "you" to refer to the user. For example, "We built the
`exec` command so you can resize a TTY session." That said, in general, try to write simple, imperative sentences that avoid the use of pronouns altogether. Say "Now, enter your SSH key" rather than "You can now enter your SSH key."

As much as possible, avoid using gendered pronouns ("he" and "she", etc.).
Either recast the sentence so the pronoun is not needed or, less preferably,
use "they" instead. If you absolutely can't get around using a gendered pronoun,
pick one and stick to it. Which one you choose is up to you. One common
convention is to use the pronoun of the author's gender, but if you prefer to
default to "he" or "she", that's fine too.

### Capitalization 

#### In general

Only proper nouns should be capitalized in body text. In general, strive to be
as strict as possible in applying this rule. Avoid using capitals for emphasis
or to denote "specialness".

The word "Docker" should always be capitalized when referring to either the
company or the technology. The only exception is when the term appears in a code
sample.

#### Starting sentences

Because code samples should always be written exactly as they would appear
on-screen, you should avoid starting sentences with a code sample.

#### In headings

Headings take sentence capitalization, meaning that only the first letter is
capitalized (and words that would normally be capitalized in a sentence, e.g.,
"Docker"). Do not use Title Case (i.e., capitalizing every word) for headings. Generally, we adhere to [AP style
for titles](http://www.quickanddirtytips.com/education/grammar/capitalizing-titles).

### Periods

We prefer one space after a period at the end of a sentence, not two. 

See [lists](#lists) below for how to punctuate list items.

### Abbreviations and acronyms

* Exempli gratia (e.g.) and id est ( i.e.): these should always have periods and
are always followed by a comma.

* Acronyms are pluralized by simply adding "s", e.g., PCs, OSs.

* On first use on a given page, the complete term should be used, with the
abbreviation or acronym in parentheses. E.g., Red Hat Enterprise Linux (RHEL).
The exception is common, non-technical acronyms like AKA or ASAP. Note that
acronyms other than i.e. and e.g. are capitalized.

* Other than "e.g." and "i.e." (as discussed above), acronyms do not take
periods, PC not P.C.


### Lists

When writing lists, keep the following in mind:

Use bullets when the items being listed are independent of each other and the
order of presentation is not important.

Use numbers for steps that have to happen in order or if you have mentioned the
list in introductory text. For example, if you wrote "There are three config
settings available for SSL, as follows:", you would number each config setting
in the subsequent list.

In all lists, if an item is a complete sentence, it should end with a
period. Otherwise, we prefer no terminal punctuation for list items.
Each item in a list should start with a capital.

### Numbers

Write out numbers in body text and titles from one to ten. From 11 on, use numerals.

### Notes

Use notes sparingly and only to bring things to the reader's attention that are
critical or otherwise deserving of being called out from the body text. Please
format all notes as follows:

    > **Note:**
    > One line of note text
    > another line of note text

### Avoid excess use of "i.e."

Minimize your use of "i.e.". It can add an unnecessary interpretive burden on
the reader. Avoid writing "This is a thing, i.e., it is like this". Just
say what it is: "This thing is …"

### Preferred usages

#### Login vs. log in. 

A "login" is a noun (one word), as in "Enter your login". "Log in" is a compound
verb (two words), as in "Log in to the terminal".

### Oxford comma

One way in which we differ from AP style is that Docker’s docs use the [Oxford
comma](http://en.wikipedia.org/wiki/Serial_comma) in all cases. That’s our
position on this controversial topic, we won't change our mind, and that’s that!

### Code and UI text styling

We require `code font` styling (monospace, sans-serif) for all text that refers
to a command or other input or output from the CLI. This includes file paths
(e.g., `/etc/hosts/docker.conf`). If you enclose text in backticks (`) markdown
will style the text as code. 

Text from a CLI should be quoted verbatim, even if it contains errors or its
style contradicts this guide. You can add "(sic)" after the quote to indicate
the errors are in the quote and are not errors in our docs.

Text taken from a GUI (e.g., menu text or button text) should appear in "double
quotes". The text should take the exact same capitalisation, etc. as appears in
the GUI. E.g., Click "Continue" to save the settings.

Text that refers to a keyboard command or hotkey is capitalized (e.g., Ctrl-D).

When writing CLI examples, give the user hints by making the examples resemble
exactly what they see in their shell: 

* Indent shell examples by 4 spaces so they get rendered as code blocks.
* Start typed commands with `$ ` (dollar space), so that they are easily
  differentiated from program output.
* Program output has no prefix.
* Comments begin with # (hash space).
* In-container shell commands, begin with `$$ ` (dollar dollar space).

Please test all code samples to ensure that they are correct and functional so
that users can successfully cut-and-paste samples directly into the CLI.

## Pull requests

The pull request (PR) process is in place so that we can ensure changes made to
the docs are the best changes possible. A good PR will do some or all of the
following:

* Explain why the change is needed
* Point out potential issues or questions
* Ask for help from experts in the company or the community
* Encourage feedback from core developers and others involved in creating the
  software being documented.

Writing a PR that is singular in focus and has clear objectives will encourage
all of the above. Done correctly, the process allows reviewers (maintainers and
community members) to validate the claims of the documentation and identify
potential problems in communication or presentation. 

### Commit messages

In order to write clear, useful commit messages, please follow these
[recommendations](http://robots.thoughtbot.com/5-useful-tips-for-a-better-commit-message).

## Links

For accessibility and usability reasons, avoid using phrases such as "click
here" for link text. Recast your sentence so that the link text describes the
content of the link, as we did in the
["Commit messages" section](#commit-messages) above.

You can use relative links (../linkeditem) to link to other pages in Docker's
documentation.

## Graphics

When you need to add a graphic, try to make the file-size as small as possible.
If you need help reducing file-size of a high-resolution image, feel free to
contact us for help.
Usually, graphics should go in the same directory as the .md file that
references them, or in a subdirectory for images if one already exists.

The preferred file format for graphics is PNG, but GIF and JPG are also
acceptable. 

If you are referring to a specific part of the UI in an image, use
call-outs (circles and arrows or lines) to highlight what you’re referring to.
Line width for call-outs should not exceed five pixels. The preferred color for
call-outs is red.

Be sure to include descriptive alt-text for the graphic. This greatly helps
users with accessibility issues.

Lastly, be sure you have permission to use any included graphics.