#!/usr/bin/perl 
use strict;
use warnings;

sub say {
    my $w = shift;
    print $w;
    print "\n";
}

sub dump {
   my $i = shift;
   &say(sprintf("Sbox[%d] = _Sbox[%d]", $i, $i));
   &say(sprintf("InvSbox[%d] = _InvSbox[%d]", $i, $i));
   &say(sprintf("Xtime2Sbox[%d] = _Xtime2Sbox[%d]", $i, $i));
   &say(sprintf("Xtime3Sbox[%d] = _Xtime3Sbox[%d]", $i, $i));
   &say(sprintf("Xtime2[%d] = _Xtime2[%d]", $i, $i));
   &say(sprintf("Xtime9[%d] = _Xtime9[%d]", $i, $i));
   &say(sprintf("XtimeB[%d] = _XtimeB[%d]", $i, $i));
   &say(sprintf("XtimeD[%d] = _XtimeD[%d]", $i, $i));
   &say(sprintf("XtimeE[%d] = _XtimeE[%d]", $i, $i));
}

for (my $i=0;$i<256;$i++) {
    &dump($i);
}



