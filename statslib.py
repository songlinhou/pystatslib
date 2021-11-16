import scipy.stats as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class CI:
    def one_proportion(self, n, phat, conf):
        z_val = st.norm.ppf(1 - (1 - conf) / 2) # this is two-side
        # z_val = st.norm.ppf(1 - (1 - conf)) # this is one-side
        se = np.sqrt(phat*(1-phat)/n)
        print('z-value=', z_val, "se=", se)
        ci = (phat - z_val * se, phat + z_val * se)
        print(ci)

    def one_proportion_conserv(self, n, phat, conf):
        # we center at phat, and try to estimate the margin of errors
        # use normal-dist (z-value)
        z_val = st.norm.ppf(1 - (1 - conf) / 2) # this is two-side
        # z_val = st.norm.ppf(1 - (1 - conf)) # this is one-side

        se = 1 / (2 * np.sqrt(n))
        print('z-value=', z_val, "se=", se)
        ci = (phat - z_val * se, phat + z_val * se)
        print(ci)

    def two_proportions(self, pos_num1, pos_num2, total_num1, total_num2, conf):
        m1, m2 = pos_num1, pos_num2 # positive numbers
        n1, n2 = total_num1, total_num2 # total numbers

        p1, p2 = m1 / n1, m2 / n2
        phat = p1 - p2
        # phat *= -1
        z_val = st.norm.ppf(1 - (1 - conf) / 2) # this is two-side
        se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        print('z-value=', z_val, "se=", se)
        print(f'{phat} +/- {z_val * se}')
        ci = (phat - z_val * se, phat + z_val * se)
        print(ci)

    def one_mean(self, mu, n, sd, conf):
        dof = n - 1
        se = sd / np.sqrt(n)
        t = st.t.ppf(1 - (1 - conf) / 2, df = dof)

        print('t-value=', t, "se=", se)
        print(f'{mu} +/- {t * se}')
        ci = (mu - t * se, mu + t * se)
        print(ci)

    def two_mean_paired(self, mu_diff, sd, n, conf):
        dof = n - 1
        mu = mu_diff
        se = sd / np.sqrt(n)
        t = st.t.ppf(1 - (1 - conf) / 2, df = dof)
        print('t-value=', t, "se=", se)
        print(f'{mu} +/- {t * se}')
        ci = (mu - t * se, mu + t * se)
        print(ci)

        if ci[0] <= 0 <= ci[1]:
            print('0 is included. maybe no difference')
        else:
            print('0 is NOT included. some difference')

    def two_means_independent_unpooled(self, mu1, mu2, sd1, sd2, n1, n2, conf):
        dof = np.min([n1 - 1, n2 - 1])
        mu_hat = mu1 - mu2
        se = np.sqrt(sd1**2 / n1 + sd2**2 / n2)
        t = st.t.ppf(1 - (1 - conf) / 2, df = dof)
        print('DOF=', dof)
        print('t-value=', t, "se=", se)

        print(f'{mu_hat} +/- {t * se}')
        ci = (mu_hat - t * se, mu_hat + t * se)
        print(ci)

    def two_means_independent_pooled(self, mu1, mu2, sd1, sd2, n1, n2, conf):
        dof = n1 + n2 - 2
        mu_hat = mu1 - mu2
        se = np.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2)/ (n1 + n2 - 2)) * np.sqrt(1/n1 + 1/n2)
        t = st.t.ppf(1 - (1 - conf) / 2, df = dof)
        print('DOF=', dof)
        print('t-value=', t, "se=", se)

        print(f'{mu_hat} +/- {t * se}')
        ci = (mu_hat - t * se, mu_hat + t * se)
        print(ci)


class HT:
    def one_proportion_two_sides(self, p0, phat, n, alpha=0.05):
        # check for assumption
        if (n * p0 >= 10 and n * (1 - p0) >= 10):
            print('sample size is large enough')
        else:
            print('sample size is NOT large enough')

        se = np.sqrt(p0 * (1 - p0) / n)
        z = abs(phat - p0) / se
        print(f'z-value is {z}')
        print(f'which means our observed sample proportion is {z} null SE above our hypothesized population proportion ABS value')
        p_val = (1 - st.norm.cdf(z)) * 2
        print(f'p-value is {p_val}')

        print(f'reject if p_value {p_val} < alpha {alpha}')
        if p_val < alpha:
            print('Reject!')
        else:
            print('Cannot reject.')

    def one_proportion_phat_larger(self, p0, phat, n, alpha=0.05):
        # check for assumption
        if (n * p0 >= 10 and n * (1 - p0) >= 10):
            print('sample size is large enough')
        else:
            print('sample size is NOT large enough')

        se = np.sqrt(p0 * (1 - p0) / n)
        z = (phat - p0) / se
        print(f'z-value is {z}')
        print(f'which means our observed sample proportion is {z} null SE above our hypothesized population proportion')
        p_val = 1 - st.norm.cdf(z)
        print(f'p-value is {p_val}')

        print(f'reject if p_value {p_val} < alpha {alpha}')
        if p_val < alpha:
            print('Reject!')
        else:
            print('Cannot reject.')

    def one_proportion_phat_smaller(self, p0, phat, n, alpha=0.05):
        # check for assumption
        if (n * p0 >= 10 and n * (1 - p0) >= 10):
            print('sample size is large enough')
        else:
            print('sample size is NOT large enough')

        se = np.sqrt(p0 * (1 - p0) / n)
        z = (p0 - phat) / se
        print(f'z-value is {z}')
        print(f'which means our observed sample proportion is {z} null SE above our hypothesized population proportion')
        p_val = 1 - st.norm.cdf(z)
        print(f'p-value is {p_val}')

        print(f'reject if p_value {p_val} < alpha {alpha}')
        if p_val < alpha:
            print('Reject!')
        else:
            print('Cannot reject.')

    def two_proportion_two_sides(self, pos1_num, pos2_num, n1, n2, alpha=0.05):
        m1, m2 = pos1_num, pos2_num # positive numbers
        # check for assumption
        phat = (m1 + m2) / (n1 + n2)
        print('phat=', phat)
        if (n1 * phat >= 10 and n1 * (1 - phat) >= 10 and n2 * phat >= 10 and n2 * (1 - phat) >= 10):
            print('sample size is large enough')
        else:
            print('sample size is NOT large enough, should not use this method')

        p1, p2 = m1 / n1, m2 / n2
        se = np.sqrt(phat * (1 - phat) * (1 / n1 + 1 / n2))
        z = (p1 - p2 - 0) / se
        print(f'z-stat is {z}')

        p_val = (1 - st.norm.cdf(abs(z))) * 2
        print(f'p-value is {p_val}')

        print(f'reject if p_value {p_val} < alpha {alpha}')
        if p_val < alpha:
            print('Reject!')
        else:
            print('Cannot reject.')

    def two_proportion_pos1_larger(self, pos1_num, pos2_num, n1, n2, alpha=0.05):
        m1, m2 = pos1_num, pos2_num # positive numbers
        # check for assumption
        phat = (m1 + m2) / (n1 + n2)
        print('phat=', phat)
        if (n1 * phat >= 10 and n1 * (1 - phat) >= 10 and n2 * phat >= 10 and n2 * (1 - phat) >= 10):
            print('sample size is large enough')
        else:
            print('sample size is NOT large enough, should not use this method')

        # p1, p2 = m1 / n1, m2 / n2
        p1, p2 = 0.52, 0.35
        # se = np.sqrt(phat * (1 - phat) * (1 / n1 + 1 / n2))
        se = 0.0338
        z = (p1 - p2 - 0) / se
        print(f'p1={p1} and p2={p2}')
        print(f'z-stat is {z}')
        # assert z > 0, "p1 > p2"

        p_val = (1 - st.norm.cdf(abs(z)))
        print(f'p-value is {p_val}')

        print(f'reject if p_value {p_val} < alpha {alpha}')
        if p_val < alpha:
            print('Reject!')
        else:
            print('Cannot reject.')

    def two_proportion_pos1_smaller(self, pos1_num, pos2_num, n1, n2, alpha=0.05):
        m1, m2 = pos1_num, pos2_num # positive numbers
        # check for assumption
        phat = (m1 + m2) / (n1 + n2)
        print('phat=', phat)
        if (n1 * phat >= 10 and n1 * (1 - phat) >= 10 and n2 * phat >= 10 and n2 * (1 - phat) >= 10):
            print('sample size is large enough')
        else:
            print('sample size is NOT large enough, should not use this method')

        p1, p2 = m1 / n1, m2 / n2
        se = np.sqrt(phat * (1 - phat) * (1 / n1 + 1 / n2))
        z = (p2 - p1 - 0) / se
        print(f'p1={p1} and p2={p2}')
        print(f'z-stat is {z}')
        # assert z > 0, "p1 > p2"

        p_val = (1 - st.norm.cdf(abs(z)))
        print(f'p-value is {p_val}')

        print(f'reject if p_value {p_val} < alpha {alpha}')
        if p_val < alpha:
            print('Reject!')
        else:
            print('Cannot reject.')

    def one_mean_two_sides(self, mu0, mu_hat, n, sd, alpha=0.05):
        dof = n - 1
        se = sd / np.sqrt(n)
        t = (mu_hat - mu0) / se
        print(f't-stat is {t}')
        p_val = (1 - st.t.cdf(abs(t), df = dof)) * 2
        print(f'p-value is {p_val}')

        print(f'reject if p_value {p_val} < alpha {alpha}')
        if p_val < alpha:
            print('Reject!')
        else:
            print('Cannot reject.')

        # confidence
        conf = 1 - alpha
        t_conf = st.t.ppf(1 - (1 - conf) / 2, df=dof)
        ci = (mu_hat - t_conf * se, mu_hat + t_conf * se)
        print('0 exist in the CI?')
        print(f'CI with {conf} confidence level = ', ci)

    def one_mean_mu_hat_larger(self, mu0, mu_hat, n, sd, alpha=0.05):
        dof = n - 1
        se = sd / np.sqrt(n)
        t = (mu_hat - mu0) / se
        print(f't-stat is {t}')
        p_val = (1 - st.t.cdf(abs(t), df = dof)) * 1
        print(f'p-value is {p_val}')

        print(f'reject if p_value {p_val} < alpha {alpha}')
        if p_val < alpha:
            print('Reject!')
        else:
            print('Cannot reject.')

        # confidence
        conf = 1 - alpha
        t_conf = st.t.ppf(1 - (1 - conf), df=dof)
        ci = (mu_hat - t_conf * se, mu_hat + t_conf * se)
        print('0 exist in the CI?')
        print(f'CI with {conf} confidence level = ', ci)

    def one_mean_mu_hat_smaller(self, mu0, mu_hat, n, sd, alpha=0.05):
        dof = n - 1
        se = sd / np.sqrt(n)
        t = (mu0 - mu_hat) / se
        print(f't-stat is {t}')
        p_val = (1 - st.t.cdf(abs(t), df = dof)) * 1
        print(f'p-value is {p_val}')

        print(f'reject if p_value {p_val} < alpha {alpha}')
        if p_val < alpha:
            print('Reject!')
        else:
            print('Cannot reject.')

        # confidence
        conf = 1 - alpha
        t_conf = st.t.ppf(1 - (1 - conf), df=dof)
        ci = (mu_hat - t_conf * se, mu_hat + t_conf * se)
        print('0 exist in the CI?')
        print(f'CI with {conf} confidence level = ', ci)

    def two_means_paired_two_sides(self, mu, sd, n, alpha=0.05):
        dof = n - 1
        se = sd / np.sqrt(n)
        t = (mu - 0) / se
        p_val = (1 - st.t.cdf(t, df= dof)) * 2
        print(f't-val = ', t)
        print(f'Our observed mean difference is {t} (estimated) SE above our null value of 0')
        print(f'reject if p_value {p_val} < alpha {alpha}')
        if p_val < alpha:
            print('Reject!')
        else:
            print('Cannot reject.')

        # confidence
        conf = 1 - alpha
        t_conf = st.t.ppf(1 - (1 - conf)/2, df=dof)
        ci = (mu - t_conf * se, mu + t_conf * se)
        print('0 exist in the CI?')
        print(f'CI with {conf} confidence level = ', ci)

    def two_means_independent_unpooled(self, mu1, mu2, sd1, sd2, n1, n2, alpha=0.05):
        dof = np.min([n1-1, n2-1])
        se = np.sqrt(sd1**2 / n1 + sd2**2 / n2)
        # se = 11.8831
        t = abs((mu1 - mu2) / se) # pay attention here
        print('dof=', dof)
        print('t-value=', t, 'se=', se)
        p_val = (1 - st.t.cdf(t, df = dof)) * 2 # if two sides
        # p_val = (1 - st.t.cdf(t, df = dof))  # if one sides
        print(f'reject if p_value {p_val} < alpha {alpha}')
        if p_val < alpha:
            print('Reject!')
        else:
            print('Cannot reject.')

        # using CI (two side)
        conf = 1 - alpha
        t_conf = st.t.ppf(1 - (1 - conf)/2, df=dof)
        ci = (mu1 - mu2 - t_conf * se, mu1 - mu2 + t_conf * se)
        print('0 exist in the CI?')
        print(f'CI with {conf} confidence level = ', ci)

    def two_means_independent_pooled(self, mu1, mu2, sd1, sd2, n1, n2, alpha=0.05):
        dof = n1 + n2 - 2
        sp = np.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2)/(n1 + n2 - 2))
        se = sp * np.sqrt(1/n1 + 1/n2)
        t = abs((mu1 - mu2) / se) # pay attention here
        print('dof=', dof)
        print('t-value=', t, 'se=', se)
        p_val = (1 - st.t.cdf(t, df = dof)) * 2 # if two sides
        # p_val = (1 - st.t.cdf(t, df = dof))  # if one sides
        print(f'reject if p_value {p_val} < alpha {alpha}')
        if p_val < alpha:
            print('Reject!')
        else:
            print('Cannot reject.')

        # using CI (two side)
        conf = 1 - alpha
        t_conf = st.t.ppf(1 - (1 - conf)/2, df=dof)
        ci = (mu1 - mu2 - t_conf * se, mu1 - mu2 + t_conf * se)
        print('0 exist in the CI?')
        print(f'CI with {conf} confidence level = ', ci)

    def chi_squared_goodness_of_fit(self, h0_seq, comp_seq, alpha=0.05):
        assert np.min(h0_seq) >=5, "each expected ount is at least 5"
        assert len(h0_seq) == len(comp_seq), "Two sequence should be equal in length"
        assert np.sum(h0_seq) == np.sum(comp_seq), "Two sequence should be equal in total"
        chi2 = np.sum((comp_seq - h0_seq) ** 2 / h0_seq)
        dof = len(h0_seq) - 1
        p_val = 1 - st.chi2.cdf(chi2, df = dof)
        print(f"chi2={chi2}\nDOF={dof}\nPval={p_val}")
        if p_val < alpha:
            print('Reject!')
        else:
            print('Cannot reject.')


    def chi_squared_homogeneity(self, df, alpha=0.05):
        col_sum = df.sum(axis = 0)
        row_sum = df.sum(axis = 1)
        df_sum = df.values.sum()

        df_vis = df.copy()
        df_vis['total'] = row_sum
        df_vis.loc['total'] = col_sum
        df_vis.iloc[-1,-1] = df_sum
        # df_vis

        df_expected = df.copy()

        if 1 in df.shape:
            for i in range(df.shape[0]):
                for j in range(df.shape[1]):
                    df_expected.iloc[i,j] = df_sum / np.multiply(*df.shape)
        else:
            for i in range(df.shape[0]):
                for j in range(df.shape[1]):
                    # df_expected.iloc[i,j] = (row_sum[i] / df_sum) * (col_sum[j] / df_sum) * df_sum
                    df_expected.iloc[i,j] = (row_sum[i] / df_sum) * col_sum[j]

        # df_expected

        if np.all(df_expected.values.flatten() >= 5):
            print('assumption is ok: every expected value is at least 5')
        else:
            print('assumption is NOT ok')

        if 1 in df.shape:
            ddof = len(df.values.flatten()) - 1
        else:
            ddof = (df.shape[0] - 1) * (df.shape[1] - 1)
        df_e_flat = df_expected.values.flatten()
        df_flat = df.values.flatten()
        print(f'DOF = {ddof}')
        chi2 = np.sum((df_flat - df_e_flat) **2 / (df_e_flat))
        print(f'chi2 = {chi2}')

        p_val = 1 - st.chi2.cdf(chi2, df = ddof)
        print(f'p-value = {p_val}')

        print(f'reject if p_value {p_val} < alpha {alpha}')
        if p_val < alpha:
            print('Reject!')
        else:
            print('Cannot reject.')
        return {'vis': df_vis, 'expected': df_expected}

    def chi_squared_test_of_association(self, df, alpha=0.05):
        return self.chi_squared_homogeneity(df, alpha)


    def ANOVA_means_of_multi_sequences(self,list_seqs, check_for_requirements=True, alpha=0.5):
        """
        test at least one pair of seqs is different in means
        """
        if check_for_requirements:
            print("1) check if sample size large enough")
            if np.min([len(s) for s in list_seqs]) >= 30:
                print("CLT applicable")
            print("2) check for normally distributed of each population")
            for id,seq in enumerate(list_seqs):
                plt.hist(seq,label=f'seq{id+1}')
            plt.show()
            print("3) check for closeness of standard deviations")
            sds = np.array([np.std(s,ddof=1) for s in list_seqs])
            if max(sds) / min(sds) < 2:
                print("passed")
            else:
                print("High standard deviation difference")
        # list_seqs = [array1, array2 ....]
        all_items = []

        for seq in list_seqs:
            all_items += list(seq)

        all_items = np.array(all_items)
        avg_total = all_items.mean()
        SSTotal = np.sum((all_items - all_items.mean()) ** 2)
        list_seq_means = np.array([np.mean(s) for s in list_seqs])
        list_seq_counts = np.array([len(s) for s in list_seqs])

        # SSGroups = np.sum([list_seq_counts[i] * (list_seq_means[i] - avg_total) ** 2 ])
        SSGroups = 0
        for i in range(len(list_seqs)):
            SSGroups += list_seq_counts[i] * (list_seq_means[i] - avg_total)**2

        SSError = 0

        for seq in list_seqs:
            SSError += np.sum((np.array(seq) - np.mean(seq)) ** 2)

        numer_dof = (len(list_seqs) - 1)
        denum_dof = (len(all_items) - len(list_seqs))
        MSGroups = SSGroups / numer_dof
        MSError = SSError / denum_dof
        print(f"Numerator dof={numer_dof}\nDenominator dof={denum_dof}")
        F = MSGroups / MSError
        print(f"F={F}")
        p = 1 - st.f.cdf(F, numer_dof, denum_dof)
        print(f"p_val={p}")
        if p < alpha:
            print('Reject!')
        else:
            print('Cannot reject.')
        return p