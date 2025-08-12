"""
Fisher Market with Cobb-Douglas Utility
MOSEK Fusion 实现（修正维度问题）
"""
from mosek.fusion import *
import numpy as np
import sys
import mosek
import collections
from collections import defaultdict
from fisher import FisherData, load_fisher_ces_data

def create_Fisher_CES_Solve(FisherData:FisherData, solve: bool = True, tol: float = 1e-4, 
                            max_iter:int = None, time_limit:float = 3600.0):
    """
    创建 Fisher Market 的 MOSEK Fusion 模型
    :param FisherData: FisherData 对象，包含问题数据
    :param solve: 是否立即求解模型
    :return: MOSEK Fusion 模型对象
    """
    # 从 FisherData 对象中提取数据
    m = FisherData.row_dim
    n = FisherData.col_dim
    budget = FisherData.w
    supply = FisherData.b
    alpha = FisherData.u
    power = FisherData.power
    x0 = FisherData.x0
    row_ptr = FisherData.row_ptr
    col_ind = FisherData.col_ind
    nnz = FisherData.nnz
    with Model('Fisher_CobbDouglas') as M:
        #set accuracy
        

        x = M.variable('x', nnz, Domain.greaterThan(0.0))
        #initialize x with x0
        if x0 is not None:
            x.setLevel(x0.reshape(nnz).tolist())
        u = M.variable('u', m, Domain.greaterThan(0.0))     
        power_x = M.variable('power_x', nnz, Domain.greaterThan(0.0))                     
        log_u = M.variable('log_u', m, Domain.unbounded()) 


        # for i in range(n):
        #     M.constraint(f'supply_{i}',
        #                 Expr.sum(x.slice([0,i], [m,i+1])),
        #                 Domain.equalsTo(supply[i]))
        # Deal Sparse Structure
        buyer2idx = defaultdict(list)   # 买家 -> [弧下标]
        for j in range(nnz):
            buyer = col_ind[j]          # 这条弧属于哪个买家
            buyer2idx[buyer].append(j)

        for i in range(n):
            if buyer2idx[i]:            # 买家 i 有弧
                M.constraint(f'supply_{i}',
                            Expr.sum(x.pick(buyer2idx[i])),
                            Domain.equalsTo(supply[i]))
            else:
                M.constraint(f'supply_{i}',
                            Expr.constTerm(0.0),
                            Domain.equalsTo(supply[i]))
                    
        for ij in range(nnz):
                M.constraint(f'power_x_{ij}',
                            Expr.vstack(x.index(ij), 1.0, power_x.index(ij)),
                            Domain.inPPowerCone(power))

        for i in range(m):
            # M.constraint(f'util_{i}',
            #             Expr.sub(u.index(i),
            #                     Expr.dot(alpha[i,:], power_x.slice([i,0], [i+1,n]))),
            #             Domain.equalsTo(0.0))
            M.constraint(f'util_{i}',
                        Expr.sub(u.index(i),
                                Expr.dot(alpha[row_ptr[i]:row_ptr[i+1]], power_x.slice([row_ptr[i]], [row_ptr[i+1]]))),
                        Domain.equalsTo(0.0))
        

        for i in range(m):
            M.constraint(f'log_x_{i}',
                        Expr.vstack(u.index(i), 1.0, log_u.index(i)),
                        Domain.inPExpCone())


        M.objective('obj', ObjectiveSense.Maximize, Expr.dot(budget, log_u))    

        M.setLogHandler(sys.stdout)
        if solve:
            M.setSolverParam("intpntCoTolRelGap", tol)
            M.setSolverParam("intpntCoTolPfeas", tol)
            M.setSolverParam("intpntCoTolDfeas", tol)
            M.setSolverParam("intpntCoTolMuRed", tol)
            M.setSolverParam("optimizerMaxTime", time_limit)
            if max_iter is not None:
                M.setSolverParam("intpntMaxIterations", max_iter)
            M.solve()
        result_dict = {}
        # ---------- 结果输出 ----------
        if M.getPrimalSolutionStatus() == SolutionStatus.Optimal:
            print('--Successfully solved the Fisher Market problem--')
            solving_time = M.getSolverDoubleInfo("optimizerTime")
            print('Solver time:', solving_time, 'seconds')
        else:
            print('Failed, Status:', M.getProblemStatus())
        result_dict['solverTime'] = M.getSolverDoubleInfo("optimizerTime")
        result_dict['status'] = M.getPrimalSolutionStatus()
        return result_dict
if __name__ == "__main__":
    import pandas as pd
    import os
    from collections import defaultdict
    result_dict = {}
    file_dir = '/home/sevan/cuLBFGSB.jl/file_dir/problem/'
    problem_dir_list = os.listdir(file_dir)
    sorted_problem_dir_list = sorted(problem_dir_list, key=lambda x: int(x.split('_')[2]))
    print(sorted_problem_dir_list)
    for problem_dir_item in sorted_problem_dir_list:
        result_dict[problem_dir_item] = {}
        print(f'Processing problem directory: {problem_dir_item}')
        problem_dir = os.path.join(file_dir, problem_dir_item)
        for meta_file_dir_item in os.listdir(problem_dir):
            meta_file_dir = os.path.join(problem_dir, meta_file_dir_item)
            if os.path.isdir(meta_file_dir):
                print(f'Processing {meta_file_dir_item}...')
                fisher_data = load_fisher_ces_data(meta_file_dir)
                model_dict = create_Fisher_CES_Solve(fisher_data, solve=True)
                result_dict[problem_dir_item][meta_file_dir_item] = {
                    'optimizerTime': model_dict['solverTime'],
                    'status': model_dict['status']
                }
    # 将结果保存为DataFrame并输出
    result_df = pd.DataFrame.from_dict({(i,j): result_dict[i][j] 
                                         for i in result_dict.keys()
                                         for j in result_dict[i].keys()},
                                        orient='index')
    result_df.index.names = ['Problem', 'MetaFileDir']
    result_df.reset_index(inplace=True)
    print(result_df)
    result_df.to_csv('fisher_ces_results1.csv', index=False)
    
    