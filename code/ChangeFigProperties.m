function ChangeFigProperties(fig,fname)
    alw = 1.5;    % AxesLineWidth
    fsz = 12;      % Fontsize
    lw = 1.2;      % LineWidth
    msz = 50;       % MarkerSize
    %%
    if exist('results','dir')~=7
        mkdir('results');
    end
    fname=['./results/' fname];
    ax=get(fig,'Children');
    set(ax, 'FontSize', fsz, 'LineWidth', alw, 'XMinorTick', 'off',...
        'YMinorTick', 'on', 'Box', 'off');
    dp = findobj(get(ax, 'Children'), 'Type', 'Line');
    dp.set('LineWidth', lw);
    dp.set('MarkerSize', msz);
    grid('on');
    print(fig, fname, '-depsc', '-r600');
%     print(fig, fname, '-dpng', '-r0');
end