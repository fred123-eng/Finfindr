# Gunicorn settings for FinFindr.
#
# gunicorn loads this file automatically from the process cwd when the
# start command is:
#   gunicorn --bind 0.0.0.0:$PORT finfindr_server:app
# which is the current Render dashboard start command for staging
# (srv-dbvrq2brjlhs73fir3a0). Command-line flags still override these
# values. A dashboard start-command change is therefore not required
# for these settings to take effect after a deploy of this commit.
# Aligning or clearing the dashboard start command remains optional
# and would be a Render config change (not done here).
#
# The d17345a Procfile already asked for gthread, 4 threads, and a 300s
# timeout, but a dashboard start command overrides the Procfile. Runtime
# logs on 2026-11-24 showed "Using worker: gthread", yet the thread count
# and timeout were not logged, so they are made explicit here.
#
# Jobs: POST /identify returns 202 and runs the work on a daemon thread.
# gunicorn threads only accept HTTP (the POST and later status polls).
# threads=8 is headroom for 5 concurrent POSTs plus polling. workers=1
# because the Render free plan is 512 MB; a second process would double
# baseline RSS.
#
# Memory risk: a sequential baseline on this plan was previously measured
# at a peak of 220.7 MB of 512 MB. Five concurrent jobs, each possibly
# running grok-4 and grok-4.7 plus a verify call, will use more. That
# 5-wide RSS figure is unmeasured. If RSS reaches 512 MB, Render may
# OOM-kill the instance. Measure 5-wide before treating it as safe.

import os

bind = "0.0.0.0:{}".format(os.environ.get("PORT", "10000"))
worker_class = "gthread"
workers = 1
threads = 8
timeout = 300
graceful_timeout = 30
keepalive = 5
worker_tmp_dir = "/dev/shm"
