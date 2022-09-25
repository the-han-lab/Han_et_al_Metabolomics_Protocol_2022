from curses import meta
import numpy as np
import pandas as pd
from collections import defaultdict
import functools
import os
import matplotlib.pyplot as plt
import shutil
from scipy import stats

class DataAnalysis:
    """
    Generates raw ion count, ISTD-corrected, and fold-change matrices, by combining the
    sample database, MS-DIAL data map, and MS-DIAL data files from each experiments across all three analytical
    methods (referred to as 'modes' below).


    Attributes
    ----------
    db : pd.DataFrame
        The sample database
    msdial_analysis_map : pd.DataFrame
        Map pointing to all MSDIAL analysis files (must be in the same directory)

        Each row points to a single worksheet in a file containing the MSDIAL analysis
        for a particular experiment, mode, and sample type
    msdial_analysis_dir : str
        Path to where the MSDIAL analysis map file lives
    cpd_library : pd.DataFrame
        Compound library containing dname to compound name mappings

        This is needed to figure out the actual metabolite / compound names corresponding to the
        dnames (e.g. m_c18n_0001 => N-ACETYLTRYPTOPHAN)
    sample_or_media_type : str
        The name of the column containing sample type (should be "sample_type") or
        media type (should be "media_type") depending on whether the data is from mammalian hosts or bacterial culture
    colonization_or_bacteria : str
        The name of the column containing colonization status (should be "colonization") or
        bacteria (should be "bacteria") depending on whether the data is from mammalian hosts or bacterial culture
    germ_free_str : str
        Should be set to "germ-free" for data from mammalian hosts,
        or "media_blank" for data from bacterial culture
    """
    
    """
    We have provided a recommended list of ISTDs for each mode, and users can change the ISTDs to include in their analysis.
    """

    ISTD_CHOICES = {
        'c18positive' : [
            'IS_2-FLUROPHENYLGLYCINE',
            'IS_4-BROMO-PHENYLALANINE',
            'IS_4-CHLORO-PHENYLALANINE',
            'IS_LEUCINE-5,5,5-D3',
            'IS_METHIONINE-METHYL-D3',
            'IS_N-BENZOYL-D5-GLYCINE',
            'IS_INDOLE-2,4,5,6,7-D5-3-ACETIC ACID',
            'IS_PHENYLALANINE-2,3,4,5,6-D5',
            'IS_TRYPTOPHAN-2,4,5,6,7-D5',
            'IS_PROGESTERONE-D9',
            'IS_D15-OCTANOIC ACID',
            'IS_D19-DECANOIC ACID',
            'IS_D27-TETRADECANOIC ACID',
            'IS_TRIDECANOIC ACID'
        ],
        'c18negative' : [
            'IS_4-BROMO-PHENYLALANINE',
            'IS_4-CHLORO-PHENYLALANINE',
            'IS_LEUCINE-5,5,5-D3',
            'IS_N-BENZOYL-D5-GLYCINE',
            'IS_INDOLE-2,4,5,6,7-D5-3-ACETIC ACID',
            'IS_PHENYLALANINE-2,3,4,5,6-D5',
            'IS_TRYPTOPHAN-2,4,5,6,7-D5',
            'IS_GLUCOSE-1,2,3,4,5,6,6-D7',
            'IS_D15-OCTANOIC ACID',
            'IS_D19-DECANOIC ACID',
            'IS_D27-TETRADECANOIC ACID',
            'IS_TRIDECANOIC ACID'
        ],
        'hilicpositive' : [
            'IS_4-BROMO-PHENYLALANINE',
            'IS_4-CHLORO-PHENYLALANINE',
            'IS_LEUCINE-5,5,5-D3',
            'IS_METHIONINE-METHYL-D3',
            'IS_INDOLE-2,4,5,6,7-D5-3-ACETIC ACID',
            'IS_PHENYLALANINE-2,3,4,5,6-D5',
            'IS_TRYPTOPHAN-2,4,5,6,7-D5',
            'IS_PROGESTERONE-D9'
        ]
    }

    '''
    Based on literature searches, the following metabolites (referred to by their dnames from the reference library)
    were removed from the final matrices, because they were unlikely to be naturally produced in mice.
    '''
    METABOLITES_TO_REMOVE = [
        # AMILORIDE
        'm_c18p_0388',
        # ATENOLOL
        'm_hilicp_0257',
        # BIS(2-ETHYLHEXYL)PHTHALATE
        'm_c18p_0521',
        # DILTIAZEM
        'm_c18p_0530',
        # ETOMIDATE
        'm_hilicp_0244',
        # FORMONONETIN
        'm_c18n_0427',
        'm_c18p_0429',
        'm_hilicp_0261',
        # ISOEUGENOL
        'm_hilicp_0144',
        # METFORMIN
        'm_c18p_0108',
        # PILOCARPINE
        'm_c18p_0364',
        # CHLORPROMAZINE
        'm_hilicp_0288',
        # CIMETIDINE
        'm_c18p_0410'
    ]


    def __init__(self,
                 db=None,
                 msdial_analysis_map=None,
                 msdial_analysis_dir=None,
                 cpd_library=None,
                 sample_or_media_type='sample_type',
                 colonization_or_bacteria='colonization',
                 germ_free_str='germ-free'):
        self.db = db
        self.msdial_analysis_map = msdial_analysis_map
        self.msdial_analysis_dir = msdial_analysis_dir
        self.cpd_library = cpd_library
        self.sample_or_media_type = sample_or_media_type
        self.colonization_or_bacteria = colonization_or_bacteria
        self.germ_free_str = germ_free_str

        self._validate_input()


    def _validate_input(self):
        if self.db is not None and self.msdial_analysis_map is not None and \
            (('collection_time' in self.msdial_analysis_map.columns and 'collection_time' not in self.db.columns) or
            ('collection_time' not in self.msdial_analysis_map.columns and 'collection_time' in self.db.columns)):
            raise Exception('collection_time should be in both or neither db and msdial_analysis_map')


    def rename_matrix(self, matrix, dname_cpd_map):
        return matrix.rename(columns=dname_cpd_map).sort_index(axis=1)


    def get_mode_from_dname(self, dname):
        if '_c18p_' in dname:
            return 'c18positive'
        elif '_c18n_' in dname:
            return 'c18negative'
        elif '_hilicp_' in dname:
            return 'hilicpositive'


    def sum_peaks(self, raw_ion_counts_matrix, dname_cpd_map):
        """
        For dname columns that correspond to the same compound (Peak 1 and 2),
        combine them into a single column with the summed raw ion counts
        """

        dnames_by_cpd = defaultdict(list)

        for dname in list(filter(lambda dname: dname in dname_cpd_map, raw_ion_counts_matrix.columns)):
            cpd = dname_cpd_map[dname]
            dnames_by_cpd[cpd].append(dname)

        for cpd, dnames in dnames_by_cpd.items():
            if len(dnames) == 1:
                continue

            #print(f'Summing peaks for {cpd} with dnames {dnames}')

            # Place the summed raw ion counts under the first dname column
            raw_ion_counts_matrix[dnames[0]] = raw_ion_counts_matrix[dnames] \
                .apply(lambda vals: np.nan if np.isnan(vals).all() else np.sum(vals), axis=1)

            # Get rid of the other dname column
            raw_ion_counts_matrix = raw_ion_counts_matrix.drop(columns=dnames[1:])

        return raw_ion_counts_matrix


    def join_msdialdf_sampledb(self, msdial_df, sample_db, exp_selection):
        # Take an MS-DIAL result and combine it with sample database.
        idxs = True
        for field, value in exp_selection.items():
            idxs = idxs & (sample_db[field] == value)

        misses = []

        # Construct map of ms_dial_sample_name values to run_id values
        sid_map = {}
        for run_id, msdial_sample_id in sample_db.loc[idxs, 'ms_dial_sample_name'].iteritems():
            if msdial_sample_id in msdial_df.columns:
                sid_map[msdial_sample_id] = run_id
            else:
                misses.append(msdial_sample_id)

        # Exclude metabolites that have been removed currently by denotation of 'x' next to the metabolite name
        if 'Remove' in msdial_df.columns:
            msdial_df = msdial_df[msdial_df['Remove'] != 'x']

        # Isolate the msdata that we want for this MS-DIAL df.
        data = msdial_df.loc[:, sid_map.keys()].values

        # The "Metabolite name" column holds the metabolite dnames
        idx = msdial_df['Metabolite name'].values
        columns = sid_map.values()

        # Get the transpose of the data so that the metabolites become columns
        msdata = pd.DataFrame(data, index=idx, columns=columns).T

        return (msdata, misses)

    def read_msdial_analyses(self, chromatography, ionization):
        """
        Read in the MSDIAL analysis worksheets specific to the current mode
        """

        filtered_runs = self.msdial_analysis_map[
            (self.msdial_analysis_map.chromatography == chromatography) &
            (self.msdial_analysis_map.ionization == ionization)
        ]

        msdata_dfs = []
        misses = []

        for idx, run in filtered_runs.iterrows():
            full_filepath = os.path.join(self.msdial_analysis_dir, run['msdial_fp'])

            exp_selection = {
                'experiment_type': run['experiment_type'],
                self.sample_or_media_type: run[self.sample_or_media_type],
                'chromatography': run['chromatography'],
                'ionization': run['ionization']
            }

            if 'collection_time' in self.msdial_analysis_map.columns:
                exp_selection['collection_time'] = run['collection_time']

            cur_msdata, cur_misses = \
                self.join_msdialdf_sampledb(pd.read_excel(io=full_filepath, sheet_name=run['sheetname'], engine='openpyxl'),
                                            self.db,
                                            exp_selection)

            msdata_dfs.append(cur_msdata)
            misses += cur_misses

        if len(msdata_dfs) == 0:
            return (pd.DataFrame(), misses)

        return (pd.concat(msdata_dfs, sort=True), misses)


    def get_matrix(self):
        """
        Read in data from MSDIAL data files
        """

        c18pos_msdata, c18pos_misses = self.read_msdial_analyses(chromatography='c18',
                                                                 ionization='positive')

        c18neg_msdata, c18neg_misses = self.read_msdial_analyses(chromatography='c18',
                                                                 ionization='negative')

        hilicpos_msdata, hilicpos_misses = self.read_msdial_analyses(chromatography='hilic',
                                                                     ionization='positive')

        db = self.db.copy(deep=True)
        db['mode'] = self.db['chromatography'] + self.db['ionization']

        all_msdata = [c18pos_msdata, c18neg_msdata, hilicpos_msdata]

        join_cols = ['experiment_type', self.sample_or_media_type, self.colonization_or_bacteria, 'sample_id']

        if 'collection_time' in self.db.columns:
            join_cols.append('collection_time')

        joined_data = functools.reduce(lambda a,b: a.join(b, how='outer'), all_msdata)
        joined_data = joined_data.join(db[join_cols], how='inner')

        joined_data.index.name='run_id'

        # Get one combined sample data row for each set of runs
        # (c18pos, c18neg, hilicpos from the same sample)
        sample_data = joined_data \
            .groupby(['sample_id']) \
            .first() \
            .join(joined_data
                .join(db[['mode']])
                .reset_index()
                # Pivot so that we know which run ids are associated with each sample
                .pivot(index='sample_id', columns='mode', values='run_id')
            )

        metadata_columns = [
            'experiment_type',
            self.sample_or_media_type,
            self.colonization_or_bacteria,
            'c18positive',
            'c18negative',
            'hilicpositive'
        ]

        if 'collection_time' in self.db.columns:
            metadata_columns.append('collection_time')

        metadata_columns = [col for col in metadata_columns if col in sample_data.columns]

        metadata = sample_data[metadata_columns]
        sample_data = sample_data.drop(columns=metadata_columns)

        return (sample_data, metadata)


    def normalize_by_istd(self, raw_ion_counts_matrix, metadata, dname_cpd_map, cpd_dname_map):
        """
        Normalize the raw ion counts within a mode and sample type based on
        internal standards to account for variations within and between experiments
        """

        istd_corrected_matrix = raw_ion_counts_matrix.copy(deep=True)

        for mode in ['c18positive', 'c18negative', 'hilicpositive']:
            dnames_in_mode = list(filter(
                lambda dname: self.get_mode_from_dname(dname) == mode,
                istd_corrected_matrix.columns.values
            ))

            istd_corrected_matrix_in_mode = istd_corrected_matrix[dnames_in_mode]
            istd_corrected_matrix_other_modes = istd_corrected_matrix.drop(columns=dnames_in_mode)

            istds = self.ISTD_CHOICES[mode]
            istd_dnames = list(map(lambda istd: cpd_dname_map[f'{istd}.{mode}'], istds))
            istd_dnames = list(set(istd_corrected_matrix_in_mode.columns.values) & set(istd_dnames))

            sampletype_dfs = []
            all_istd_sums = []

            for sample_type, sample_ids in metadata.groupby([self.sample_or_media_type]).groups.items():
                sample_type_data = istd_corrected_matrix_in_mode.loc[sample_ids]

                # Remove rows with all NaNs to account for mode-specific sample data that was removed
                istd_data = sample_type_data[istd_dnames].dropna(axis=0, how='all')

                # Drop any internal standards compound that has any nan
                # Ensures that each row is normalized consistently
                istd_data = istd_data.dropna(axis=1, how='any')

                removed_istd_dnames = list(set(istd_dnames) - set(istd_data.columns.values))

                #if removed_istd_dnames:
                    #print(f'{sample_type}, {mode}: ISTD columns removed with a NaN: {removed_istd_dnames}')
                    #print(f'{len(istd_data.columns)} of {len(istd_dnames)} columns remaining')

                # Sum the ISTD raw ion counts for each row
                istd_sums = istd_data.sum(axis=1)

                # Divide each raw ion count by the corresponding ISTD sum for the row
                sampletype_dfs.append(sample_type_data.divide(istd_sums, axis=0))

                all_istd_sums.append(istd_sums)

            all_istd_sums = pd.concat(all_istd_sums, sort=False)

            # Multiply corrected data by the median to bring it back to the original magnitude
            istd_corrected_matrix_in_mode = pd.concat(sampletype_dfs, sort=False) * all_istd_sums.median()

            istd_corrected_matrix = istd_corrected_matrix_in_mode \
                .join(istd_corrected_matrix_other_modes, how='outer')

        # Remove all internal standard columns
        istd_cols = list(filter(
            lambda dname: dname_cpd_map[dname].startswith('IS_'),
            istd_corrected_matrix.columns
        ))

        istd_corrected_matrix = istd_corrected_matrix.drop(columns=istd_cols)

        return istd_corrected_matrix.sort_index().sort_index(axis=1)


    def get_fold_change(self, istd_corrected_matrix, metadata):
        """
        Calculate the log2 of the ratio of the data to mean value of its germ-free controls
        from the same experiment and sample type
        """

        exp_sampletype_dfs = []

        istd_corrected_matrix = istd_corrected_matrix.add(1)

        for exp_sampletype, sample_ids in metadata.groupby(['experiment_type', self.sample_or_media_type]).groups.items():
            exp_sampletype_metadata = metadata.loc[sample_ids]
            germfree_metadata = exp_sampletype_metadata[exp_sampletype_metadata[self.colonization_or_bacteria] == self.germ_free_str]

            germfree_mean = istd_corrected_matrix.loc[germfree_metadata.index.values] \
                .mean(axis=0)

            cur_exp_sampletype_data = istd_corrected_matrix.loc[sample_ids] / germfree_mean

            exp_sampletype_dfs.append(np.log2(cur_exp_sampletype_data))

        return pd.concat(exp_sampletype_dfs, sort=True).sort_index()


    def collapse_replicates(self, fold_change_matrix, metadata):
        """
        Average the samples within the same experiment, sample type, and colonization
        """

        metadata_cols = ['experiment_type', self.sample_or_media_type, self.colonization_or_bacteria]

        # Make sure we average based on the raw data instead of the log values
        matrix = 2 ** fold_change_matrix
        matrix = matrix.join(metadata[metadata_cols])

        return np.log2(matrix.groupby(metadata_cols).mean())


    def remove_dnames(self, matrix):
        dnames = set(self.METABOLITES_TO_REMOVE) & set(matrix.columns)
        return matrix.drop(columns=dnames)


    def _get_cv(self, istd_corrected_matrix, metadata):
        metadata_cols = ['experiment_type', self.colonization_or_bacteria, self.sample_or_media_type]
        istd_corrected_matrix = istd_corrected_matrix.join(metadata[metadata_cols])

        def get_variation(group):
            values = group.values
            return stats.variation(values[~np.isnan(values)])

        # Calculate coefficient of variation for each set of replicates
        return istd_corrected_matrix.groupby(metadata_cols) \
            .aggregate(get_variation).sort_index(axis=1)


    def plot_cv_histograms(self, istd_corrected_matrix, metadata):
        """
        Plots histograms for coefficients of variation (all data and co-detected data across multiple modes)
        for istd-corrected values within each experiment, colonization, and sample type.

        Also generates a recommended mode picking preference file

        Parameters
        ----------
        istd_corrected_matrix : pd.DataFrame
            An istd-corrected matrix with compound names instead of dnames as columns.

            Column names should be in the format <metabolite>.<mode>
        metadata : pd.DataFrame
            The metadata table
        """
        variations = self._get_cv(istd_corrected_matrix, metadata)
        variations.to_excel('variations.xlsx')

        """
        cols_by_metabolite should look something like:

        {
            '1,5-ANHYDRO-GLUCITOL': ['1,5-ANHYDRO-GLUCITOL.c18negative'],
            '1,6-ANHYDRO-B-GLUCOSE': ['1,6-ANHYDRO-B-GLUCOSE.c18negative', '1,6-ANHYDRO-B-GLUCOSE.c18positive'],
            ...
        }
        """

        cols_by_metabolite = defaultdict(list)

        for col in variations.columns:
            if variations[col].isnull().all():
                continue

            cols_by_metabolite[col.split('.')[0]].append(col)

        self.generate_cv_histograms(variations, cols_by_metabolite)
        mode_picker_template = self.generate_cv_histograms_codetected(variations, cols_by_metabolite)

        return pd.DataFrame.from_records(mode_picker_template).set_index('metabolite')


    def generate_cv_histograms(self, variations, cols_by_metabolite):
        plt.style.use('seaborn-deep')

        dir_path = "cv_histograms"

        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)

        os.mkdir(dir_path)

        print(f"Generating histograms in {dir_path}/")

        for metabolite, cols in cols_by_metabolite.items():
            if len(cols) == 1:
                continue

            x = [variations[col].values for col in cols]
            labels = [col.split('.')[1] for col in cols]

            fig, ax = plt.subplots(1, 1)
            ax.set_title(metabolite)

            ax.hist(x, label=labels)

            ax.set_xlabel('Relative Standard Deviation')
            ax.set_ylabel('Frequency')
            ax.legend(loc='upper right')

            plt.savefig(f"{dir_path}/{metabolite}.pdf")
            plt.close()


    def generate_cv_histograms_codetected(self, variations, cols_by_metabolite):
        dir_path = "cv_histograms_codetected"

        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)

        os.mkdir(dir_path)

        print(f"Generating histograms in {dir_path}/")

        mode_picker_template = []
        suffixes_to_features = {
            'c18positive': 'c18p',
            'c18negative' : 'c18n',
            'hilicpositive' : 'hilicp',
        }

        for metabolite, cols in cols_by_metabolite.items():
            mode_picker_row = {
                'metabolite': metabolite,
                'mode_detected': ', '.join([suffixes_to_features[col.split('.')[1]] for col in cols]),
            }

            # Nothing to plot if only one mode was detected,
            if len(cols) == 1:
                mode_picker_row['mode_pref'] = suffixes_to_features[cols[0].split('.')[1]]
                mode_picker_template.append(mode_picker_row)
                continue

            # Only consider rows in which there is data for more than one mode
            filtered_variations = variations[cols][variations[cols].count(axis=1) > 1]

            x = []
            labels = []
            preferred_modes = []
            for col in cols:
                values = filtered_variations[col].values
                suffix = col.split('.')[1]

                # Skip modes that are not codetected with any other mode
                if np.isnan(values).all():
                    continue

                x.append(values)
                labels.append(suffix)

                # Only prefer modes for which the average CV is less than or equal to 0.2, users can change this threshold
                if np.mean(values) <= 0.2:
                    preferred_modes.append(suffixes_to_features[suffix])

            mode_picker_row['mode_pref'] = mode_picker_row['mode_detected'] if len(preferred_modes) == 0 else ', '.join(preferred_modes)
            mode_picker_template.append(mode_picker_row)

            # Skip over metabolites for which is there no codetected data
            if len(labels) == 0:
                continue

            fig, ax = plt.subplots(1, 1)
            ax.set_title(metabolite)

            ax.hist(x, label=labels)

            ax.set_xlabel('Relative Standard Deviation')
            ax.set_ylabel('Frequency')
            ax.legend(loc='upper right')

            plt.savefig(f"{dir_path}/{metabolite}.pdf")
            plt.close()


        return mode_picker_template


    def collapse_modes(self, fold_change_matrix, mode_picker, strict_mode=False):
        """Collapses the 3 mode columns for each metabolite
        (.c18positive, .c18negative, .hilicpositive) into a single column
        by either picking a single mode, or averaging the values between multiple modes.

        Parameters
        ----------
        fold_change_matrix : pd.DataFrame
            A fold change matrix with compound names instead of dnames as columns.

            Column names should be in the format <metabolite>.<mode>
        mode_picker : pd.DataFrame
            Mode picking definition file containing the preferred mode(s) for each metabolite.

            The preferred modes are specified in the "mode_pref" column.
        """

        def get_colname(cpd, feature):
            return cpd + '.' + features_to_suffixes[feature]

        mode_picker = mode_picker[mode_picker['mode_detected'].notnull()]

        mode_picker.index = mode_picker.index.map(lambda val: val.split('.')[0])

        features_to_suffixes = {
            'c18p': 'c18positive',
            'c18n': 'c18negative',
            'hilicp': 'hilicpositive',
        }

        for cpd, row in mode_picker.iterrows():
            #print("Processing compound {}:".format(cpd))

            features = list(map(lambda feature: feature.strip(), row['mode_detected'].split(',')))
            mode_prefs = list(map(lambda mode_pref: mode_pref.strip(), row['mode_pref'].split(',')))

            #print(f"Features: {features}; mode prefs: {mode_prefs}")

            feature_colnames = [get_colname(cpd, feature) for feature in features]
            preferred_colnames = [get_colname(cpd, pref) for pref in mode_prefs]
            remaining_colnames = [get_colname(cpd, feature) for feature in (set(features) - set(mode_prefs))]

            # If compounds have been removed from the fold change matrix as per
            # METABOLITES_TO_REMOVE, then skip over these missing columns here
            if strict_mode:
                if len(set(feature_colnames) & set(fold_change_matrix.columns)) < len(feature_colnames):
                    #print(f'Skipping {cpd} as not all features were found in the fold change matrix')
                    continue
            else:
                feature_colnames = list(set(feature_colnames) & set(fold_change_matrix.columns))
                preferred_colnames = list(set(preferred_colnames) & set(fold_change_matrix.columns))
                remaining_colnames = list(set(feature_colnames) - set(preferred_colnames))

                if len(feature_colnames) == 0:
                    #print(f'Skipping {cpd} as none of the features were found in the fold change matrix')
                    continue

                if len(preferred_colnames) == 0:
                    #print(f'No preferred modes for {cpd} were found. Using the feature columns instead.')
                    preferred_colnames = feature_colnames

            preferred_data = fold_change_matrix[preferred_colnames]

            # Average the data from the preferred modes
            fold_change_matrix[cpd] = preferred_data.apply(lambda values: np.log2(np.mean(2 ** values)), axis=1)

            if len(remaining_colnames) > 0:
                remaining_data = fold_change_matrix[remaining_colnames]

                # For any rows that have nans after extracting the data from the preferred modes,
                # average the data from the remaining modes so that we're not throwing away data.
                fold_change_matrix[cpd] = fold_change_matrix.apply(
                    lambda sample: np.log2(np.mean(2 ** remaining_data.loc[sample.name])) if pd.isna(sample[cpd]) else sample[cpd],
                    axis=1
                )

            fold_change_matrix = fold_change_matrix.drop(columns=feature_colnames)

        return fold_change_matrix.sort_index(axis=1)


    def run(self,
            output_cpd_names=False,
            remove_dnames=False,
            exps=[]):
        # Dictionary of dnames to compound names
        # (dnames with multiple compounds concatenated together)
        dname_cpd_map = self.cpd_library \
            .groupby(['dname'])['Compound'] \
            .apply(lambda compounds: ', '.join(sorted(set(compounds)))) \
            .to_dict()

        dname_cpd_map = {dname: cpd.strip() + '.' + self.get_mode_from_dname(dname) for dname, cpd in dname_cpd_map.items()}

        cpd_dname_map = {value:key for key, value in dname_cpd_map.items()}

        raw_ion_counts_matrix, all_metadata = self.get_matrix()

        if len(exps) > 0:
            all_metadata = all_metadata[all_metadata['experiment_type'].isin(exps)]
            raw_ion_counts_matrix = raw_ion_counts_matrix.loc[all_metadata.index].dropna(axis=1, how='all')

        raw_ion_counts_matrix = self.sum_peaks(raw_ion_counts_matrix, dname_cpd_map)
        istd_corrected_matrix = self.normalize_by_istd(raw_ion_counts_matrix, all_metadata, dname_cpd_map, cpd_dname_map)

        fold_change_matrix = self.get_fold_change(istd_corrected_matrix, all_metadata)

        result = {
            'raw_ion_counts_matrix': raw_ion_counts_matrix,
            'istd_corrected_matrix': istd_corrected_matrix,
            'fold_change_matrix': fold_change_matrix,
        }

        if remove_dnames:
            result = {key:self.remove_dnames(matrix) for (key,matrix) in result.items()}

        if output_cpd_names:
            result = {key:self.rename_matrix(matrix, dname_cpd_map) for (key,matrix) in result.items()}

        result['metadata'] = all_metadata

        return result
