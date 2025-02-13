FROM mambaorg/micromamba

USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    git \
 && apt-get clean -y \
 && rm -rf /var/lib/apt/lists/*

USER $MAMBA_USER

COPY --chown=$MAMBA_USER:$MAMBA_USER env.yml /tmp/env.yml
RUN micromamba install -y -n base -f /tmp/env.yml \
 && micromamba clean --all --yes

