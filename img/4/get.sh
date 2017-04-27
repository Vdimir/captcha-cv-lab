#!/bin/bash

for i in `seq 1 100`;
do

curl -A "Mozilla/5.0 (iPhone; U; CPU iPhone OS 4_3_3 like Mac OS X; en-us) AppleWebKit/533.17.9 (KHTML, like Gecko) Version/5.0.2 Mobile/8J2 Safari/6533.18.5" "http://openvibe.inria.fr/openvibe/wp-content/plugins/si-contact-form/captcha/securimage_show.php" > "${i}.jpg"


done
