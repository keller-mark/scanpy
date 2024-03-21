from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np
from sklearn.utils import check_array, check_random_state

from ... import logging as logg
from ..._compat import old_positionals
from ..._settings import settings
from ..._utils import AnyRandom, NeighborsView
from scanpy.tools._umap import prepare_umap_or_densmap

if TYPE_CHECKING:
    from anndata import AnnData

_InitPos = Literal["paga", "spectral", "random"]

@old_positionals(
    "min_dist",
    "spread",
    "n_components",
    "maxiter",
    "alpha",
    "gamma",
    "negative_sample_rate",
    "init_pos",
    "random_state",
    "a",
    "b",
    "copy",
    "neighbors_key",
)
def densmap(
    adata: AnnData,
    *,
    min_dist: float = 0.5,
    spread: float = 1.0,
    n_components: int = 2,
    maxiter: int | None = None,
    alpha: float = 1.0,
    gamma: float = 1.0,
    negative_sample_rate: int = 5,
    init_pos: _InitPos | np.ndarray | None = "spectral",
    random_state: AnyRandom = 0,
    a: float | None = None,
    b: float | None = None,
    copy: bool = False,
    neighbors_key: str | None = None,
    dens_lambda: float = 2.0,
    dens_frac: float = 0.3,
    dens_var_shift: float = 0.1,
) -> AnnData | None:
    """\
    Embed the neighborhood graph using UMAP [McInnes18]_.

    UMAP (Uniform Manifold Approximation and Projection) is a manifold learning
    technique suitable for visualizing high-dimensional data. Besides tending to
    be faster than tSNE, it optimizes the embedding such that it best reflects
    the topology of the data, which we represent throughout Scanpy using a
    neighborhood graph. tSNE, by contrast, optimizes the distribution of
    nearest-neighbor distances in the embedding such that these best match the
    distribution of distances in the high-dimensional space.  We use the
    implementation of `umap-learn <https://github.com/lmcinnes/umap>`__
    [McInnes18]_. For a few comparisons of UMAP with tSNE, see this `preprint
    <https://doi.org/10.1101/298430>`__.

    Parameters
    ----------
    adata
        Annotated data matrix.
    min_dist
        The effective minimum distance between embedded points. Smaller values
        will result in a more clustered/clumped embedding where nearby points on
        the manifold are drawn closer together, while larger values will result
        on a more even dispersal of points. The value should be set relative to
        the ``spread`` value, which determines the scale at which embedded
        points will be spread out. The default of in the `umap-learn` package is
        0.1.
    spread
        The effective scale of embedded points. In combination with `min_dist`
        this determines how clustered/clumped the embedded points are.
    n_components
        The number of dimensions of the embedding.
    maxiter
        The number of iterations (epochs) of the optimization. Called `n_epochs`
        in the original UMAP.
    alpha
        The initial learning rate for the embedding optimization.
    gamma
        Weighting applied to negative samples in low dimensional embedding
        optimization. Values higher than one will result in greater weight
        being given to negative samples.
    negative_sample_rate
        The number of negative edge/1-simplex samples to use per positive
        edge/1-simplex sample in optimizing the low dimensional embedding.
    init_pos
        How to initialize the low dimensional embedding. Called `init` in the
        original UMAP. Options are:

        * Any key for `adata.obsm`.
        * 'paga': positions from :func:`~scanpy.pl.paga`.
        * 'spectral': use a spectral embedding of the graph.
        * 'random': assign initial embedding positions at random.
        * A numpy array of initial embedding positions.
    random_state
        If `int`, `random_state` is the seed used by the random number generator;
        If `RandomState` or `Generator`, `random_state` is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
    a
        More specific parameters controlling the embedding. If `None` these
        values are set automatically as determined by `min_dist` and
        `spread`.
    b
        More specific parameters controlling the embedding. If `None` these
        values are set automatically as determined by `min_dist` and
        `spread`.
    copy
        Return a copy instead of writing to adata.
    method
        Chosen implementation.

        ``'umap'``
            Umap’s simplical set embedding.
        ``'rapids'``
            GPU accelerated implementation.

            .. deprecated:: 1.10.0
                Use :func:`rapids_singlecell.tl.umap` instead.
    neighbors_key
        If not specified, umap looks .uns['neighbors'] for neighbors settings
        and .obsp['connectivities'] for connectivities
        (default storage places for pp.neighbors).
        If specified, umap looks .uns[neighbors_key] for neighbors settings and
        .obsp[.uns[neighbors_key]['connectivities_key']] for connectivities.

    Returns
    -------
    Returns `None` if `copy=False`, else returns an `AnnData` object. Sets the following fields:

    `adata.obsm['X_umap']` : :class:`numpy.ndarray` (dtype `float`)
        UMAP coordinates of data.
    `adata.uns['umap']` : :class:`dict`
        UMAP parameters.

    """

    # Everything on the LHS of an expression is returned
    (
        adata,
        neighbors_key,
        start,
        neighbors,
        a,
        b,
        init_coords,
        random_state,
        neigh_params,
        X,
    ) = prepare_umap_or_densmap(
        adata,
        min_dist=min_dist,
        spread=spread,
        n_components=n_components,
        maxiter=maxiter,
        alpha=alpha,
        gamma=gamma,
        negative_sample_rate=negative_sample_rate,
        init_pos=init_pos,
        random_state=random_state,
        a=a,
        b=b,
        copy=copy,
        neighbors_key=neighbors_key,
    )

    from umap.umap_ import simplicial_set_embedding

    # the data matrix X is really only used for determining the number of connected components
    # for the init condition in the UMAP embedding
    default_epochs = 500 if neighbors["connectivities"].shape[0] <= 10000 else 200
    n_epochs = default_epochs if maxiter is None else maxiter

    densmap_kwds = {
        "graph_dists": neighbors["distances"],
        "n_neighbors": neighbors["params"]["n_neighbors"],
        # Default params from umap package
        # Reference: https://github.com/lmcinnes/umap/blob/868e55cb614f361a0d31540c1f4a4b175136025c/umap/umap_.py#L1692
        "lambda": dens_lambda,
        "frac": dens_frac,
        "var_shift": dens_var_shift,
    }

    X_densmap, _ = simplicial_set_embedding(
        data=X,
        graph=neighbors["connectivities"].tocoo(),
        n_components=n_components,
        initial_alpha=alpha,
        a=a,
        b=b,
        gamma=gamma,
        negative_sample_rate=negative_sample_rate,
        n_epochs=n_epochs,
        init=init_coords,
        random_state=random_state,
        metric=neigh_params.get("metric", "euclidean"),
        metric_kwds=neigh_params.get("metric_kwds", {}),
        densmap=True,
        densmap_kwds=densmap_kwds,
        output_dens=False,
        verbose=settings.verbosity > 3,
    )

    adata.obsm["X_densmap"] = X_densmap  # annotate samples with densMAP coordinates
    logg.info(
        "    finished",
        time=start,
        deep=("added\n" "    'X_densmap', densMAP coordinates (adata.obsm)"),
    )
    return adata if copy else None
