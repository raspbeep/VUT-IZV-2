#!/usr/bin/env python3.9
# coding=utf-8

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import zipfile

plt.style.use("seaborn-v0_8-whitegrid")
# select 4 specific regions
selected_regions = ["PHA", "STC", "JHC", "PLK"]


def load_data(filename: str) -> pd.DataFrame:
    """
    Load csv files from all years and months(except for pedestrians) into a single dataframe
    :param filename: path to the zip file
    :return: loaded Dataframe
    """
    headers = [
        "p1",
        "p36",
        "p37",
        "p2a",
        "weekday(p2a)",
        "p2b",
        "p6",
        "p7",
        "p8",
        "p9",
        "p10",
        "p11",
        "p12",
        "p13a",
        "p13b",
        "p13c",
        "p14",
        "p15",
        "p16",
        "p17",
        "p18",
        "p19",
        "p20",
        "p21",
        "p22",
        "p23",
        "p24",
        "p27",
        "p28",
        "p34",
        "p35",
        "p39",
        "p44",
        "p45a",
        "p47",
        "p48a",
        "p49",
        "p50a",
        "p50b",
        "p51",
        "p52",
        "p53",
        "p55a",
        "p57",
        "p58",
        "a",
        "b",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "p5a",
    ]
    empty_file_names = ["08.csv", "09.csv", "10.csv", "11.csv", "12.csv", "13.csv"]

    regions = {
        "PHA": "00",
        "STC": "01",
        "JHC": "02",
        "PLK": "03",
        "ULK": "04",
        "HKK": "05",
        "JHM": "06",
        "MSK": "07",
        "OLK": "14",
        "ZLK": "15",
        "VYS": "16",
        "PAK": "17",
        "LBK": "18",
        "KVK": "19",
    }
    # invert dictionary and add file extension
    inv_regions = {v + ".csv": k for k, v in regions.items()}

    all_data = pd.DataFrame()
    with zipfile.ZipFile(filename) as outer_zip:
        for outer_file_name in outer_zip.namelist():
            with zipfile.ZipFile(outer_zip.open(outer_file_name)) as inner_zip:
                for inner_file_name in inner_zip.namelist():
                    with inner_zip.open(inner_file_name) as inner_file:
                        if inner_file_name not in empty_file_names:
                            if inner_file_name != "CHODCI.csv":
                                data = pd.read_csv(
                                    inner_file,
                                    encoding="cp1250",
                                    sep=";",
                                    names=headers,
                                    low_memory=False,
                                )
                                # find region by value in dict
                                data["region"] = inv_regions[inner_file_name]
                                all_data = pd.concat([all_data, data])
    return all_data


def parse_data(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Parse data and cast them into float, int, datetime and categorical columns
    :param df: dataframe to parse
    :param verbose: print info about data
    :return: parsed dataframe
    """
    if verbose:
        print(f"orig_size={df.memory_usage(deep=True).sum() / 10 ** 6:.2f} MB")

    new_df = df.copy()

    integer_cols = ["p2b", "p13a", "p13b", "p13c", "p14", "p45a", "p47", "p53"]

    for col in integer_cols:
        new_df[col] = new_df[col].astype(int, errors="ignore")

    float_cols = ["d", "e"]
    for col in float_cols:
        new_df[col] = new_df[col].str.replace(",", ".")
    #     new_df[col] = new_df[col].astype('f', errors='ignore')
    new_df["d"] = new_df.d.replace("", np.nan).astype(float, errors="ignore")
    new_df["e"] = new_df.e.replace("", np.nan).astype(float, errors="ignore")
    new_df["d"] = pd.to_numeric(new_df["d"], errors="coerce")
    new_df["e"] = pd.to_numeric(new_df["e"], errors="coerce")

    new_df["p13a"] = pd.to_numeric(new_df["p13a"], errors="coerce")
    new_df["p13b"] = pd.to_numeric(new_df["p13b"], errors="coerce")
    new_df["p13c"] = pd.to_numeric(new_df["p13c"], errors="coerce")

    category_cols = [
        "p12",
        "p36",
        "p37",
        "weekday(p2a)",
        "p6",
        "p7",
        "p8",
        "p9",
        "p10",
        "p11",
        "p15",
        "p16",
        "p17",
        "p18",
        "p19",
        "p20",
        "p21",
        "p22",
        "p23",
        "p24",
        "p27",
        "p28",
        "p34",
        "p35",
        "p44",
        "p45a",
        "p48a",
        "p49",
        "p50a",
        "p50b",
        "p51",
        "p52",
        "p55a",
        "p57",
        "p58",
        "a",
        "b",
        "h",
        "i",
        "j",
        "k",
        "l",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "p5a",
        "f",
        "g",
        "n",
    ]
    for col in category_cols:
        new_df[col] = new_df[col].astype("category")

    new_df["date"] = pd.to_datetime(new_df["p2a"]).astype("datetime64[ns]")
    new_df = new_df.drop_duplicates("p1")

    if verbose:
        print(f"new_size={new_df.memory_usage(deep=True).sum() / 10 ** 6:.2f} MB")

    return new_df


def plot_visibility(
    df: pd.DataFrame, fig_location: str = None, show_figure: bool = False
):
    """
    Plot graph of accidents by daytime(day-night) and visibility(good-bad) for each region
    :param df: dataframe with data
    :param fig_location: location to save figure
    :param show_figure: show figure in window
    :return: None
    """
    new_df = df[df["region"].isin(selected_regions)]
    sns.set_style("whitegrid", {"axes.grid": False})
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharey="col")
    fig.suptitle("Počet nehod podle denního času a viditelnosti", fontsize=16)
    for *_, subplot in np.ndenumerate(axes):
        subplot.set_ylim(0, 110_000)
    ax = axes.flatten()
    sns.set_palette("pastel")
    new_df = new_df.groupby(["region", "p19"])["p1"].count().unstack().reset_index()

    new_df["night_ok"] = new_df[4] + new_df[6]
    new_df["night_bad"] = new_df[5] + new_df[7]
    new_df["day_ok"] = new_df[1]
    new_df["day_bad"] = new_df[2] + new_df[3]

    # night ok
    ax0 = sns.barplot(
        data=new_df, ax=ax[0], x="region", y="night_ok", order=selected_regions
    )
    ax[0].set_title("Viditelnost: v noci - nezhorsena")
    ax[0].set(xlabel=None, ylabel="Pocet nehod")
    ax0.bar_label(ax0.containers[0])
    # night bad
    ax1 = sns.barplot(
        data=new_df, ax=ax[1], x="region", y="night_bad", order=selected_regions
    )
    ax[1].set_title("Viditelnost: v noci - zhorsena")
    ax[1].set(xlabel=None)
    ax[1].set(ylabel=None)
    ax1.bar_label(ax1.containers[0])
    # day ok
    ax2 = sns.barplot(
        data=new_df, ax=ax[2], x="region", y="day_ok", order=selected_regions
    )
    ax[2].set_title("Viditelnost: ve dne - nezhorsena")
    ax[2].set(xlabel="kraj", ylabel="Pocet nehod")
    ax2.bar_label(ax2.containers[0])
    # day bad
    ax3 = sns.barplot(
        data=new_df, ax=ax[3], x="region", y="day_bad", order=selected_regions
    )
    ax[3].set_title("Viditelnost: ve dne - zhorsena")
    ax[3].set(ylabel=None, xlabel="kraj")
    ax3.bar_label(ax3.containers[0])

    fig.tight_layout()

    if fig_location is not None:
        plt.savefig(fig_location)
    if show_figure:
        plt.show()
    plt.close()

    sns.set_style("whitegrid", {"axes.grid": True})


def plot_direction(
    df: pd.DataFrame, fig_location: str = None, show_figure: bool = False
):
    """
    Plot graph of accidents by direction of movement for each region
    :param df: dataframe with data
    :param fig_location: location to save figure
    :param show_figure: show figure in window
    :return: None
    """
    new_df = df[df["region"].isin(selected_regions)]
    new_df = new_df[new_df["p7"] != 0]
    new_df["p7"] = new_df.p7.astype(str)

    new_df["month"] = new_df["date"].dt.month
    new_df["sum"] = 1
    # merge 2(bocni) and 3(z boku) into one category `bocni`
    new_df["p7"] = new_df["p7"].map(
        {"1": "celni", "2": "bocni", "3": "bocni", "4": "zezadu"}
    )
    new_df = new_df.groupby(["region", "p7", "month"]).agg({"sum": "sum"}).reset_index()
    g = sns.catplot(
        data=new_df,
        x="month",
        y="sum",
        col="region",
        col_order=selected_regions,
        hue="p7",
        hue_order=["bocni", "zezadu", "celni"],
        kind="bar",
        errorbar=None,
        col_wrap=2,
        palette="muted",
        sharex=False,
    )
    g.set_titles("Kraj: {col_name}")
    g.fig.suptitle("Pocet nehod podla smeru jazdy a mesiaca", fontsize=16)

    sns.move_legend(g, "center right", title="Druh srazky")
    count = 0
    for ax in g.axes:
        ax.set(xlabel="Mesic", ylabel="Pocet nehod")
        ax.yaxis.get_label().set_visible(True)
        if count < 2:
            ax.set_xlabel("")
        if count % 2 == 1:
            ax.set_ylabel("")
        count += 1

    g.tight_layout()
    if fig_location is not None:
        plt.savefig(fig_location)
    if show_figure:
        plt.show()
    plt.close()

    pass


def plot_consequences(
    df: pd.DataFrame, fig_location: str = None, show_figure: bool = False
):
    """
    Plot graph of accidents by consequences subsampled to months for each region
    :param df: dataframe with data
    :param fig_location: location to save figure
    :param show_figure: show figure in window
    :return: None
    """
    new_df = df.copy()
    new_df = new_df[new_df["region"].isin(selected_regions)]

    # create column with consequences
    new_df.loc[new_df["p13c"] > 0, "lehke"] = 1
    new_df.loc[new_df["p13b"] > 0, "tezke"] = 1
    new_df.loc[new_df["p13a"] > 0, "smrt"] = 1
    # remove consequences if there are any of higher severity (lehke < tezke < smrt)
    new_df.loc[new_df["tezke"] > 0, "lehke"] = 0
    new_df.loc[new_df["smrt"] > 0, "tezke"] = 0
    new_df.loc[new_df["smrt"] > 0, "Nasledky"] = "usmrceni"
    new_df.loc[new_df["tezke"] > 0, "Nasledky"] = "tezke zraneni"
    new_df.loc[new_df["lehke"] > 0, "Nasledky"] = "lehke zraneni"
    # drop accidents with no consequences
    new_df = new_df[new_df["Nasledky"].notna()]
    new_df = new_df[["date", "region", "Nasledky", "lehke", "tezke", "smrt"]]
    new_df = new_df.pivot_table(
        index=["date", "region"],
        values=["lehke", "tezke", "smrt"],
        aggfunc={"lehke": np.sum, "tezke": np.sum, "smrt": np.sum},
        fill_value=0,
    )

    # subsample to months
    new_df = (
        new_df.groupby(["region"])
        .resample("MS", level=0)
        .sum(numeric_only=True)
        .reset_index()
    )
    new_df = pd.melt(
        new_df,
        id_vars=["date", "region"],
        value_vars=["lehke", "tezke", "smrt"],
        var_name="Nasledky",
        value_name="pocet",
    )

    g = sns.FacetGrid(
        new_df,
        col="region",
        hue="Nasledky",
        col_wrap=2,
        col_order=selected_regions,
        palette="muted",
        sharex=False,
        height=4,
        aspect=1.5,
        sharey=True,
    )
    g = g.map(sns.lineplot, "date", "pocet").add_legend(title="Nasledky")
    g.fig.suptitle("Pocet nehod podla mesiacov v jednotivych krajoch", fontsize=16)
    g.set_titles("Kraj: {col_name}")
    count = 0
    for ax in g.axes.flat:
        ax.set(xlabel="Mesic", ylabel="Pocet nehod")
        x_ticks = ax.get_xticks()
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(
            [pd.to_datetime(tm, unit="d").strftime(r"%m/%y") for tm in x_ticks]
        )
        ax.set_xlim(np.array(["2016-01-01", "2022-01-02"], dtype="datetime64[D]"))
        ax.set_xlabel("")
        if count % 2 == 1:
            ax.set_ylabel("")
        ax.set_axisbelow(True)
        ax.set_facecolor("whitesmoke")
        ax.grid(color="white", linestyle="-")
        for key, spine in ax.spines.items():
            spine.set_visible(False)

        count += 1

    g.tight_layout()

    if fig_location is not None:
        plt.savefig(fig_location)
    if show_figure:
        plt.show()
    plt.close()


if __name__ == "__main__":
    data_frame = load_data("data/data.zip")
    data_frame_2 = parse_data(data_frame, True)
    plot_visibility(data_frame_2, "01_visibility.png", False)
    plot_direction(data_frame_2, "02_direction.png", False)
    plot_consequences(data_frame_2, "03_consequences.png", True)
