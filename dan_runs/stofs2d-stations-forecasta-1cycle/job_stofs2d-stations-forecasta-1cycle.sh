#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --job-name=stofs2d-stations-forecasta-1cycle
#SBATCH --output=job_stofs2d-stations-forecasta-1cycle.out
#SBATCH --error=job_stofs2d-stations-forecasta-1cycle.err
# Memory: the laptop run of the fields nowcast job peaked at 2.0 GiB, so no
# memory request is set and the cluster default is used. Only if this job
# dies out of memory, remove one '#' from the next line so it reads #SBATCH
# (see dan_runs/README_adcirc_validation.md, Troubleshooting).
##SBATCH --mem=16G

cd /lustre/code/devNG/dev-Next-Gen-NOS-OFS-Skill-Assessment
export PYTHONPATH=$PWD/src
export PROJ_NETWORK=ON
# forecast_a: -s gives the cycle date and -f the cycle hour. create_1dplot
# replaces -e with the cycle time + 180 h (get_fcst_dates in
# src/ofs_skill/model_processing/get_fcst_cycle.py), which is the -e below.
CMD=(python ./bin/visualization/create_1dplot.py \
    -c ./dan_runs/stofs2d-stations-forecasta-1cycle/stofs2d-stations-forecasta-1cycle.conf \
    -p ./dan_runs/stofs2d-stations-forecasta-1cycle \
    -o stofs_2d_glo -s 2026-10-14T00:00:00Z -e 2026-10-21T12:00:00Z -f 00z -d MSL -ws forecast_a -t stations -vs water_level -so co-ops)

echo "Branch:  $(git rev-parse --abbrev-ref HEAD)"
echo "Commit:  $(git rev-parse HEAD)"
echo "Start:   $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Command: ${CMD[*]}"
echo

# /usr/bin/time -v writes peak memory ("Maximum resident set size", in KB)
# and elapsed time to the end of job_stofs2d-stations-forecasta-1cycle.err.
if [ -x /usr/bin/time ]; then
    /usr/bin/time -v "${CMD[@]}"
else
    echo "/usr/bin/time not found: peak memory and elapsed time are not recorded."
    "${CMD[@]}"
fi
status=$?

echo
echo "End:     $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Exit code: $status"
exit $status
