word_indices  = processEmail('abe above abil ab tip the cow');
fprintf('here\n');
length(word_indices)

x = emailFeatures(word_indices)

%==== Processed Email ====

%ab abov abil ab tip the cow

%=========================
%word_indices =

 %     2
  %    6
  %    3
  %    2
  % 1695
  % 1666