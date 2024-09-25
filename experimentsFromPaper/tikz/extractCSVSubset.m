

loadDirs = {'../results_traffic', '../results_shuttle', '../results_hyperspectral'};

for i = 1:length(loadDirs)
    for name = {'Z', 'I', 'W', 'C'}
        T  = readtable([loadDirs{i},'/',name{1},'/results.csv']);
        writetable(T(T.k == 5,:), [loadDirs{i},'/',name{1},'/results_k5.csv']);
        
        for p = [10,50,100,120]
            T  = readtable([loadDirs{i},'/',name{1},'/results.csv']);
            writetable(T(T.p == p,:), [loadDirs{i},'/',name{1},'/results_p',num2str(p),'.csv']);
        end
    end
end

