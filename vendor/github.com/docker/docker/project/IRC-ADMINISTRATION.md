# Freenode IRC Administration Guidelines and Tips

This is not meant to be a general "Here's how to IRC" document, so if you're
looking for that, check Google instead. ♥

If you've been charged with helping maintain one of Docker's now many IRC
channels, this might turn out to be useful.  If there's information that you
wish you'd known about how a particular channel is organized, you should add
deets here! :)

## `ChanServ`

Most channel maintenance happens by talking to Freenode's `ChanServ` bot.  For
example, `/msg ChanServ ACCESS <channel> LIST` will show you a list of everyone
with "access" privileges for a particular channel.

A similar command is used to give someone a particular access level.  For
example, to add a new maintainer to the `#docker-maintainers` access list so
that they can contribute to the discussions (after they've been merged
appropriately in a `MAINTAINERS` file, of course), one would use `/msg ChanServ
ACCESS #docker-maintainers ADD <nick> maintainer`.

To setup a new channel with a similar `maintainer` access template, use a
command like `/msg ChanServ TEMPLATE <channel> maintainer +AV` (`+A` for letting
them view the `ACCESS LIST`, `+V` for auto-voice; see `/msg ChanServ HELP FLAGS`
for more details).

## Troubleshooting

The most common cause of not-getting-auto-`+v` woes is people not being
`IDENTIFY`ed with `NickServ` (or their current nickname not being `GROUP`ed with
their main nickname) -- often manifested by `ChanServ` responding to an `ACCESS
ADD` request with something like `xyz is not registered.`.

This is easily fixed by doing `/msg NickServ IDENTIFY OldNick SecretPassword`
followed by `/msg NickServ GROUP` to group the two nicknames together.  See
`/msg NickServ HELP GROUP` for more information.
