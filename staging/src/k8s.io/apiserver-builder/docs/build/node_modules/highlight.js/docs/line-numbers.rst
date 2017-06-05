Line numbers
============

Highlight.js' notable lack of line numbers support is not an oversight but a
feature. Following is the explanation of this policy from the current project
maintainer (hey guys!):

    One of the defining design principles for highlight.js from the start was
    simplicity. Not the simplicity of code (in fact, it's quite complex) but
    the simplicity of usage and of the actual look of highlighted snippets on
    HTML pages. Many highlighters, in my opinion, are overdoing it with such
    things as separate colors for every single type of lexemes, striped
    backgrounds, fancy buttons around code blocks and — yes — line numbers.
    The more fancy stuff resides around the code the more it distracts a
    reader from understanding it.

    This is why it's not a straightforward decision: this new feature will not
    just make highlight.js better, it might actually make it worse simply by
    making it look more bloated in blog posts around the Internet. This is why
    I'm asking people to show that it's worth it.

    The only real use-case that ever was brought up in support of line numbers
    is referencing code from the descriptive text around it. On my own blog I
    was always solving this either with comments within the code itself or by
    breaking the larger snippets into smaller ones and describing each small
    part separately. I'm not saying that my solution is better. But I don't
    see how line numbers are better either. And the only way to show that they
    are better is to set up some usability research on the subject. I doubt
    anyone would bother to do it.

    Then there's maintenance. So far the core code of highlight.js is
    maintained by only one person — yours truly. Inclusion of any new code in
    highlight.js means that from that moment I will have to fix bugs in it,
    improve it further, make it work together with the rest of the code,
    defend its design. And I don't want to do all this for the feature that I
    consider "evil" and probably will never use myself.

This position is `subject to discuss <http://groups.google.com/group/highlightjs>`_.
Also it doesn't stop anyone from forking the code and maintaining line-numbers implementation separately.
