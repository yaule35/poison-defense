service nginx restart

/usr/local/bin/uwsgi --ini uwsgi.ini

tail -f /var/log/nginx/error.log